"""Experiment class for Monash transfer learning with deep learning models.

Trains any TSLib model on Monash TSF data (univariate, enc_in=1) with per-window
RevIN applied in the training/test loops. Tests on a target benchmark CSV
(channel-independent).

Usage via run.py:
    python run.py --task_name monash_transfer --model DLinear \
        --data Monash --root_path /root/monash_tsf_all/data \
        --target_data ETTh1 --target_root_path /root/dataset/ETT-small/ \
        --target_data_path ETTh1.csv \
        --seq_len 96 --pred_len 96 --enc_in 1 --dec_in 1 --c_out 1 ...
"""
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Monash_Transfer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Monash_Transfer, self).__init__(args)

    def _build_model(self):
        # Models check task_name for forward routing; use 'long_term_forecast'
        # so existing models (DLinear, PatchTST, etc.) use their forecast path.
        import copy
        model_args = copy.copy(self.args)
        model_args.task_name = 'long_term_forecast'
        model = self.model_dict[self.args.model](model_args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        from data_provider.data_loader_monash import Dataset_Monash, Dataset_Monash_Test

        args = self.args
        size = [args.seq_len, args.label_len, args.pred_len]

        if flag in ('train', 'val'):
            data_set = Dataset_Monash(
                args=args,
                root_path=args.root_path,
                flag=flag,
                size=size,
            )
            shuffle = (flag == 'train')
        else:
            # Test: use target dataset (channel-independent)
            target_root = getattr(args, 'target_root_path', None) or args.root_path
            target_file = getattr(args, 'target_data_path', None) or args.data_path
            data_set = Dataset_Monash_Test(
                args=args,
                root_path=target_root,
                flag='test',
                size=size,
                data_path=target_file,
            )
            shuffle = False

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            drop_last=(flag == 'train'),
        )
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    @staticmethod
    def _apply_revin(batch_x, batch_y, pred_len):
        """Per-window RevIN: normalize by input window's mean/std.

        Args:
            batch_x: (B, seq_len, 1)
            batch_y: (B, label_len + pred_len, 1)
            pred_len: int

        Returns:
            batch_x_norm, batch_y_norm, mu, std
        """
        # Compute stats from input (batch_x)
        mu = batch_x.mean(dim=1, keepdim=True)   # (B, 1, 1)
        std = batch_x.std(dim=1, keepdim=True)    # (B, 1, 1)
        std = torch.clamp(std, min=1e-8)

        batch_x_norm = (batch_x - mu) / std
        batch_y_norm = (batch_y - mu) / std

        return batch_x_norm, batch_y_norm, mu, std

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Apply RevIN
                batch_x, batch_y, mu, std = self._apply_revin(
                    batch_x, batch_y, self.args.pred_len
                )

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # Forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Apply RevIN normalization
                batch_x, batch_y, mu, std = self._apply_revin(
                    batch_x, batch_y, self.args.pred_len
                )

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # Forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y_target = batch_y[:, -self.args.pred_len:, :]
                        loss = criterion(outputs, batch_y_target)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y_target = batch_y[:, -self.args.pred_len:, :]
                    loss = criterion(outputs, batch_y_target)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Apply RevIN to input
                mu = batch_x.mean(dim=1, keepdim=True)
                std = batch_x.std(dim=1, keepdim=True)
                std = torch.clamp(std, min=1e-8)
                batch_x_norm = (batch_x - mu) / std

                # Normalize batch_y for decoder input (label part)
                batch_y_norm = (batch_y - mu) / std

                # Decoder input
                dec_inp = torch.zeros_like(batch_y_norm[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_norm[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # Forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x_norm, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x_norm, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]  # (B, pred_len, 1)

                # De-RevIN: transform predictions back to StandardScaler domain
                outputs = outputs * std + mu

                # Ground truth (already in StandardScaler domain)
                batch_y_true = batch_y[:, -self.args.pred_len:, :]

                pred = outputs.detach().cpu().numpy()
                true = batch_y_true.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # Result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        f = open("result_monash_transfer.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mse, mae
