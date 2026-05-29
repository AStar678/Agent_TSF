import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        print(f"x1.shape: {x.shape}")
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        print(f"x2.shape: {x.shape}")
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        print(f"x.shape: {x.shape}")
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        print(f"x.shape: {x.shape}")
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars




import torch
import torch.nn as nn
import math

# 补齐依赖项：PositionalEmbedding (使用你上一个问题中的结构)
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 注意这里的 x 传进来时，形状是 [Batch * Variates, Num_Patches, d_model]
        # x.size(1) 实际上获取的是 Num_Patches
        return self.pe[:, :x.size(1)]

# 你的原版 PatchEmbedding
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars











import torch
import torch.nn as nn
import torch.nn.functional as F
# 假设你的 Embed.py 在同级目录下
from Embed import PatchEmbedding, TokenEmbedding 

class DynamicRouter(nn.Module):
    """
    轻量级路由器：根据输入的全局统计特性（均值、标准差等）输出混合权重
    """
    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(c_in, c_in // reduction),
            nn.ReLU(),
            nn.Linear(c_in // reduction, 2), # 输出两个权重的 logit：[Patch权重, Global权重]
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: [bs, n_vars, seq_len]
        b, c, s = x.shape
        y = self.avg_pool(x).squeeze(-1) # [bs, n_vars]
        weights = self.mlp(y) # [bs, 2]
        return weights

class DynamicHybridEmbedding(nn.Module):
    def __init__(self, c_in, d_model, patch_len, stride, padding, dropout=0.1):
        super().__init__()
        # 1. Patch-wise 分支 (保留 PatchTST 的核心优势)
        self.patch_embedding = PatchEmbedding(
            d_model=d_model, patch_len=patch_len, 
            stride=stride, padding=padding, dropout=dropout
        )
        
        # 2. Variate-wise 分支 (捕捉变量间的全局交互)
        # 这里使用 TokenEmbedding 提取点级特征，再通过线性层映射
        self.global_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        # 3. 动态路由器
        self.router = DynamicRouter(c_in=c_in)
        
        # 维度对齐层：将全局特征对齐到 Patch 的数量级
        self.align_layer = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [bs, seq_len, n_vars] -> 原生输入格式
        """
        # 转换维度以适应 PatchEmbedding 和 Router: [bs, n_vars, seq_len]
        x_enc = x.transpose(1, 2)
        bs, n_vars, seq_len = x_enc.shape
        
        # 计算动态权重
        weights = self.router(x_enc) # [bs, 2]
        w_patch = weights[:, 0].view(bs, 1, 1, 1)
        w_global = weights[:, 1].view(bs, 1, 1, 1)

        # --- 分支 A: Patch Embedding ---
        # PatchEmbedding 需要转置后的输入 [bs, n_vars, seq_len]
        # out_patch: [bs * n_vars, patch_num, d_model]
        out_patch, _ = self.patch_embedding(x_enc)
        out_patch = out_patch.view(bs, n_vars, -1, out_patch.shape[-1]) # [bs, n_vars, patch_num, d_model]

        # --- 分支 B: Global Variate-wise Embedding ---
        # ⚠️ 修复点 1：TokenEmbedding 需要原生输入 x [bs, seq_len, n_vars]
        out_global = self.global_embedding(x) # 输出形状: [bs, seq_len, d_model]
        
        # ⚠️ 修复点 2：在 seq_len (dim=1) 上求均值，提取全局特征
        out_global = out_global.mean(dim=1)   # 形状变为: [bs, d_model]
        
        out_global = self.align_layer(out_global) # 映射维度
        # 广播到与 Patch 相同的形状 [bs, n_vars, patch_num, d_model]
        out_global = out_global.view(bs, 1, 1, -1).expand_as(out_patch)

        # --- 动态融合 ---
        # 最终输出: [bs * n_vars, patch_num, d_model]
        out = (w_patch * out_patch) + (w_global * out_global)
        return out.view(-1, out.shape[2], out.shape[3])

def simulate_call():
    # 1. 参数设置
    bs, seq_len, n_vars = 32, 96, 7
    d_model = 128
    patch_len, stride = 16, 8
    
    print(f"输入数据形状: [Batch:{bs}, Seq_Len:{seq_len}, Vars:{n_vars}]")
    
    # 2. 构造模拟输入
    # 模拟两类数据：一类是有规律的周期信号，一类是纯随机噪声
    x = torch.randn(bs, seq_len, n_vars)
    
    # 3. 实例化动态混合 Embedding
    hybrid_emb = DynamicHybridEmbedding(
        c_in=n_vars, d_model=d_model, 
        patch_len=patch_len, stride=stride, padding=stride
    )
    
    # 4. 前向传播
    output = hybrid_emb(x)
    
    print("\n--- 输出结果 ---")
    print(f"混合 Embedding 输出形状: {output.shape}") 
    # 预期: [bs * n_vars, patch_num, d_model]
    
    # 5. 观察路由器的行为
    with torch.no_grad():
        x_enc = x.transpose(1, 2)
        weights = hybrid_emb.router(x_enc)
        print(f"前 3 个样本的动态权重 (Patch vs Global):")
        for i in range(3):
            print(f" Sample {i}: Patch={weights[i,0]:.4f}, Global={weights[i,1]:.4f}")

if __name__ == "__main__":
    simulate_call()