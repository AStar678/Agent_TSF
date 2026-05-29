import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        print(x_enc.shape)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        print(enc_out.shape)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]


        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
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





def test_model():
    # 1. 构造模型配置 (Configs)
    # 根据你的要求和代码中的调用，模拟一个 argparse 的配置类
    class Configs:
        def __init__(self):
            self.task_name = 'long_term_forecast'  # 设定为长序列预测任务
            self.seq_len = 96                      # 输入序列长度
            self.pred_len = 336                    # 预测序列长度
            self.enc_in = 7                        # 变量/通道数
            self.d_model = 512                     # Transformer 隐藏层维度
            self.embed = 'timeF'                   # 时间特征嵌入方式
            self.freq = 'h'                        # 时间频率
            self.dropout = 0.1                     # Dropout 比例
            self.factor = 1                        # Attention factor
            self.n_heads = 8                       # 多头注意力头数
            self.d_ff = 2048                       # FFN 层维度
            self.activation = 'gelu'               # 激活函数
            self.e_layers = 2                      # Encoder 的层数
            self.num_class = 3                     # 分类任务类别数（仅为防止其他分支报错）

    configs = Configs()

    # 2. 初始化模型
    print("正在初始化模型...")
    model = Model(configs)
    model.eval() # 设置为评估模式

    # 3. 构造虚拟输入数据 (Dummy Inputs)
    batch_size = 32
    seq_len = 96
    pred_len = 336
    channels = 7
    mark_dim = 4 # 时间协变量特征维度 (例如: 年、月、日、时)

    # 编码器输入: [Batch, seq_len, channels] -> [32, 96, 7]
    x_enc = torch.randn(batch_size, seq_len, channels)
    
    # 编码器时间特征输入: [Batch, seq_len, mark_dim] -> [32, 96, 4]
    x_mark_enc = torch.randn(batch_size, seq_len, mark_dim)

    # 解码器输入及时间特征 (在此 iTransformer 结构的 forecast 方法中未被实际计算使用，但需占位传参)
    x_dec = torch.randn(batch_size, pred_len, channels)
    x_mark_dec = torch.randn(batch_size, pred_len, mark_dim)

    # 4. 执行前向传播
    print("\n--- 维度信息 ---")
    print(f"输入 x_enc 维度: {x_enc.shape}  [Batch, Seq_Len, Channels]")
    
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"输出 output 维度: {output.shape} [Batch, Pred_Len, Channels]")
    
    # 验证输出维度是否符合预期 (32, 336, 7)
    assert output.shape == (batch_size, pred_len, channels), "输出维度不符合预期！"
    print("\n✅ 测试通过：模型成功输出了预测结果！")

if __name__ == '__main__':
    # 确保当前目录下有 layers 文件夹 (包含 Transformer_EncDec, SelfAttention_Family, Embed)
    test_model()