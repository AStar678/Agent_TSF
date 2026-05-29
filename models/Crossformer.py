import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from PatchTST import FlattenHead


from math import ceil


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2
        self.task_name = configs.task_name

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(configs.d_model, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(configs, 1 if l == 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,
                            1, configs.dropout,
                            self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l), configs.factor
                            ) for l in range(configs.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(configs, (self.pad_out_len // self.seg_len), configs.factor, configs.d_model, configs.n_heads,
                                           configs.d_ff, configs.dropout),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    self.seg_len,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    # activation=configs.activation,
                )
                for l in range(configs.e_layers + 1)
            ],
        )
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))  # [32, 7, 96]
        print(f"enc_value_embedding output shape: {x_enc.shape}")
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)

        return dec_out

    def anomaly_detection(self, x_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))

        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # Output from Non-stationary Transformer
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None



if __name__ == "__main__":
    print("="*50)
    print("CrossFormer 模型初始化与维度流向测试...")

    # 1. 伪造 configs 对象
    class Config:
        def __init__(self):
            self.task_name = 'long_term_forecast' # 可选: imputation, anomaly_detection, classification
            self.seq_len = 96        # 输入序列长度
            self.pred_len = 720      # 预测序列长度
            self.enc_in = 7          # 特征通道数 (变量数)
            
            # 模型特有结构参数
            self.d_model = 128       # 隐藏层维度
            self.d_ff = 256          # FFN 层扩展维度
            self.n_heads = 4         # 注意力头数
            self.e_layers = 3        # Encoder 层数 (决定了 Router 的合并次数)
            self.dropout = 0.1
            self.factor = 10         # Router 机制的路由因子
            
            # 分类任务占位参数 (防止切换 task_name 时报错)
            self.num_class = 3       

    configs = Config()

    # 2. 实例化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(configs).to(device)

    # 3. 构造模拟输入张量
    batch_size = 32
    mark_dim = 4 # 时间协变量维度

    # 时序模型标准输入输出字典 (CrossFormer 的 forecast 方法实际上只用到了 x_enc)
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in).to(device)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, mark_dim).to(device)
    x_dec = torch.randn(batch_size, configs.pred_len, configs.enc_in).to(device)
    x_mark_dec = torch.randn(batch_size, configs.pred_len, mark_dim).to(device)

    print(f"\n[输入数据]")
    print(f"x_enc -> {x_enc.shape}  # [Batch, seq_len, enc_in]")

    # 4. 执行前向传播
    model.eval()
    with torch.no_grad():
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    # 5. 验证输出
    print(f"\n[输出数据]")
    print(f"out -> {out.shape}  # [Batch, pred_len, enc_in]")
    
    # 预测任务的维度断言
    if configs.task_name in ['long_term_forecast', 'short_term_forecast']:
        assert out.shape == (batch_size, configs.pred_len, configs.enc_in)
        print("✅ 预测任务测试通过！")
    print("="*50)