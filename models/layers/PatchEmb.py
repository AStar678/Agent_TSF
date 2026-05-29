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
        print(x.shape)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        print(x.shape)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        print(x.shape)
        x = self.value_embedding(x) + self.position_embedding(x)
        print(x.shape)
        return self.dropout(x), n_vars

if __name__ == "__main__":
    print("="*50)
    print("PatchEmbedding 模块维度流向测试...")

    # 1. 设定超参数
    batch_size = 32
    n_vars = 7           # 变量/通道数
    seq_len = 96         # 原始序列长度
    patch_len = 16       # 每个 Patch 的长度
    stride = 8           # 滑动窗口的步长 (重叠 8 个点)
    padding = stride     # 尾部填充长度 (通常设为 stride 保证能切尽)
    d_model = 128        # Transformer 的隐藏层维度

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 实例化模型
    patch_emb = PatchEmbedding(d_model=d_model, patch_len=patch_len, 
                               stride=stride, padding=padding, dropout=0.1).to(device)

    # 3. 构造输入数据
    # ⚠️ 注意：这里的输入期望是 [Batch, Variates, Time]
    x = torch.randn(batch_size, n_vars, seq_len).to(device)
    
    print(f"\n[1. 原始输入]")
    print(f"x -> {x.shape}  # [Batch, Variates, Seq_Len]")

    # 4. 执行前向传播
    patch_emb.eval()
    with torch.no_grad():
        out, out_vars = patch_emb(x)

    # 计算预期的 Patch 数量
    # 公式: num_patches = (seq_len + padding - patch_len) / stride + 1
    expected_patches = int((seq_len + padding - patch_len) / stride + 1)

    print(f"\n[2. 经过切块 (Patching) 与映射后的输出]")
    print(f"out -> {out.shape}  # [Batch * Variates, Num_Patches, d_model]")
    print(f"out_vars -> {out_vars} (原封不动返回的变量数)")
    print(f"自动计算出的 Patch 数量: {expected_patches}")
    
    # 断言检查
    assert out.shape == (batch_size * n_vars, expected_patches, d_model)
    assert out_vars == n_vars
    print("="*50)
    print("✅ 测试通过：时间序列被成功切块，并融合了变量维度。")