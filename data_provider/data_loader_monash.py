"""PyTorch Dataset classes for Monash transfer learning.

- Dataset_Monash: loads Monash TSF files for training/validation (univariate, enc_in=1)
- Dataset_Monash_Test: loads target CSV channel-independently for testing (univariate, enc_in=1)

Both datasets return raw data (no RevIN); per-window RevIN is applied in the
experiment class (exp_monash_transfer.py) during forward pass.
"""
import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from monash_utils.data import (
    parse_tsf, load_monash,
    MONASH_DIR, DEFAULT_MONASH_FILES, PRESET_FILESETS, DEFAULT_MIN_STD,
)


class Dataset_Monash(Dataset):
    """Monash TSF dataset for training/validation.

    Samples univariate sliding windows from all loaded Monash series.
    Each series is globally z-scored (per-series), then windows are extracted.
    Returns shape (seq_len, 1) for seq_x, (label_len + pred_len, 1) for seq_y.
    """

    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='', target='OT',
                 scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        self.args = args
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.flag = flag
        self.scale = scale
        self.root_path = root_path  # monash_dir

        # Monash-specific args
        self.monash_files = getattr(args, 'monash_files', None)
        self.monash_preset = getattr(args, 'monash_preset', 'etth_like')
        self.monash_max_per_series = getattr(args, 'monash_max_per_series', 200)
        self.val_ratio = getattr(args, 'monash_val_ratio', 0.1)
        self.seed = getattr(args, 'seed', 2021)

        self.__read_data__()

    def __read_data__(self):
        """Load Monash TSF files and build per-series z-scored sliding windows."""
        files = self.monash_files
        if files is None:
            files = PRESET_FILESETS.get(self.monash_preset, DEFAULT_MONASH_FILES)

        series_list = load_monash(self.root_path, files, verbose=True)

        W = self.seq_len + self.pred_len
        stride = max(1, self.pred_len // 2)

        rng = np.random.default_rng(self.seed)

        # Build windows with per-series global z-score
        all_windows = []
        for s in series_list:
            if s.size < W + 1:
                continue
            mu, sd = s.mean(), s.std()
            if sd < DEFAULT_MIN_STD:
                continue
            s_n = ((s - mu) / sd).astype(np.float32)

            starts = np.arange(0, s.size - W, stride, dtype=np.int64)
            if starts.size == 0:
                continue
            if starts.size > self.monash_max_per_series:
                starts = rng.choice(starts, size=self.monash_max_per_series, replace=False)

            for st in starts:
                all_windows.append(s_n[st: st + W])

        rng.shuffle(all_windows)
        total = len(all_windows)
        val_size = max(1, int(total * self.val_ratio))

        if self.flag == 'val':
            windows = all_windows[:val_size]
        else:
            windows = all_windows[val_size:]

        # Stack into (N, W) array
        self.data = np.stack(windows, axis=0)  # (N, seq_len + pred_len)
        print(f"[Dataset_Monash] flag={self.flag}, windows={self.data.shape[0]}, "
              f"total_series={len(series_list)}")

    def __getitem__(self, index):
        # window layout: [0..seq_len) = input, [seq_len-label_len..seq_len+pred_len) = target
        win = self.data[index]  # (W,)

        s_end = self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = win[:s_end].reshape(-1, 1)           # (seq_len, 1)
        seq_y = win[r_begin:r_end].reshape(-1, 1)    # (label_len + pred_len, 1)

        # Zero time marks (Monash has no consistent time stamps)
        seq_x_mark = np.zeros((self.seq_len, 4), dtype=np.float32)
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 4), dtype=np.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, data):
        return data  # no global scaler; RevIN handled externally


class Dataset_Monash_Test(Dataset):
    """Target CSV dataset for testing in Monash transfer.

    Loads a benchmark CSV (e.g. ETTh1.csv) and returns each channel as an
    independent univariate sample. StandardScaler is fit on the train split
    following Time-MoE protocol.
    """

    def __init__(self, args, root_path, flag='test', size=None,
                 features='S', data_path='ETTh1.csv', target='OT',
                 scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        self.args = args
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        """Load target CSV, apply StandardScaler, build channel-independent windows."""
        import pandas as pd

        csv_path = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(csv_path)
        n_rows = len(df)

        # Determine borders (Time-MoE protocol)
        base = os.path.basename(csv_path).lower()
        if "etth" in base:
            b1s = [0, 12*30*24 - self.seq_len, 12*30*24 + 4*30*24 - self.seq_len]
            b2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
        elif "ettm" in base:
            b1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4 + 4*30*24*4 - self.seq_len]
            b2s = [12*30*24*4, 12*30*24*4 + 4*30*24*4, 12*30*24*4 + 8*30*24*4]
        else:
            num_train = int(n_rows * 0.7)
            num_test = int(n_rows * 0.2)
            num_vali = n_rows - num_train - num_test
            b1s = [0, num_train - self.seq_len, n_rows - num_test - self.seq_len]
            b2s = [num_train, num_train + num_vali, n_rows]

        # Load all numeric columns (skip 'date')
        cols = df.columns[1:]
        values = df[cols].values.astype(np.float32)  # (T, V)
        self.n_channels = values.shape[1]

        # Fit scaler on train segment only
        self.scaler = StandardScaler()
        train_seg = values[b1s[0]:b2s[0]]
        self.scaler.fit(train_seg)

        # Get test segment
        test_seg = values[b1s[2]:b2s[2]]
        if self.scale:
            scaled_test = self.scaler.transform(test_seg).astype(np.float32)
        else:
            scaled_test = test_seg

        # Build channel-independent windows
        # scaled_test: (T_test, V)
        T_test = scaled_test.shape[0]
        W = self.seq_len + self.pred_len

        # For each channel, slide windows
        all_windows = []
        for c in range(self.n_channels):
            seq = scaled_test[:, c]  # (T_test,)
            n_win = T_test - W + 1
            for st in range(n_win):
                all_windows.append(seq[st: st + W])

        self.data = np.stack(all_windows, axis=0)  # (N, W)
        print(f"[Dataset_Monash_Test] file={self.data_path}, channels={self.n_channels}, "
              f"windows={self.data.shape[0]}")

    def __getitem__(self, index):
        win = self.data[index]  # (W,)

        s_end = self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = win[:s_end].reshape(-1, 1)           # (seq_len, 1)
        seq_y = win[r_begin:r_end].reshape(-1, 1)    # (label_len + pred_len, 1)

        seq_x_mark = np.zeros((self.seq_len, 4), dtype=np.float32)
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 4), dtype=np.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, data):
        return data  # metrics computed in StandardScaler domain
