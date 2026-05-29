import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x



if __name__ == "__main__":
    print("="*50)
    print("正在初始化测试环境...")

    # ==========================================
    # 1. 构造一个占位的“模拟注意力层” (Mock Attention)
    # 真实场景中，这里应该是 ProbSparseAttention 或 FullAttention
    # ==========================================
    class MockAttention(nn.Module):
        def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
            # Attention 的核心输出维度必须与 queries 保持一致
            B, L_q, D = queries.shape
            # 伪造一个输出张量 (通常是 values 的加权求和)
            out = torch.randn(B, L_q, D).to(queries.device)
            # 伪造一个注意力权重矩阵 (用于可视化或提取，很多模型不需要所以随便给个维度)
            attn = torch.zeros(B, 1, L_q, keys.shape[1]).to(queries.device)
            return out, attn

    # ==========================================
    # 2. 设定超参数
    # ==========================================
    batch_size = 16
    d_model = 64      # 隐藏层特征维度
    d_ff = 256        # FFN层扩展维度
    seq_len = 96      # Encoder 输入序列长度
    pred_len = 24     # Decoder 预测长度
    label_len = 48    # Decoder 引导序列长度 (通常前一半是真实历史，后一半是预测位)
    dec_in_len = label_len + pred_len # Decoder 的总输入长度 (72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 3. 实例化网络层
    # ==========================================
    # 3.1 单独测试一下 ConvLayer (下采样层)
    conv_layer = ConvLayer(c_in=d_model).to(device)

    # 3.2 构造 Encoder (包含 2个 Attention 层 和 1个 Conv 下采样层)
    attn_layers = [
        EncoderLayer(MockAttention(), d_model, d_ff),
        EncoderLayer(MockAttention(), d_model, d_ff)
    ]
    conv_layers = [ConvLayer(c_in=d_model)]
    encoder = Encoder(attn_layers, conv_layers, norm_layer=nn.LayerNorm(d_model)).to(device)

    # 3.3 构造 Decoder (包含 1个 Decoder 层)
    dec_layers = [
        DecoderLayer(self_attention=MockAttention(), cross_attention=MockAttention(), d_model=d_model, d_ff=d_ff)
    ]
    # 假设最终我们要把特征维度映射回单变量预测 (out_channels = 1)
    projection = nn.Linear(d_model, 1) 
    decoder = Decoder(dec_layers, norm_layer=nn.LayerNorm(d_model), projection=projection).to(device)

    # ==========================================
    # 4. 构造模拟张量并执行前向传播
    # ==========================================
    # Encoder 输入: [Batch, Seq_Len, d_model]
    enc_inputs = torch.randn(batch_size, seq_len, d_model).to(device)
    # Decoder 输入: [Batch, label_len + pred_len, d_model]
    dec_inputs = torch.randn(batch_size, dec_in_len, d_model).to(device)

    print("\n[单独测试 ConvLayer 下采样]")
    print(f"输入维度 -> {enc_inputs.shape}")
    conv_out = conv_layer(enc_inputs)
    # 计算逻辑: padding=2, kernel=3 的 conv 后长度约等于原长，然后 stride=2 的 MaxPool 让长度减半
    print(f"经过 ConvLayer 输出维度 -> {conv_out.shape} (注意序列长度被折半了)")

    print("\n[测试完整的 Encoder (编码器)]")
    # 结构: Attn -> Conv -> Attn
    encoder.eval()
    with torch.no_grad():
        print(f"Encoder 输入维度 -> {enc_inputs.shape}")
        enc_out, attns = encoder(enc_inputs)
    print(f"Encoder 最终输出维度 -> {enc_out.shape} (长度被中间的 Conv 层折半)")
    print(f"收集到的 Attention 矩阵数量 -> {len(attns)} 个")

    print("\n[测试完整的 Decoder (解码器)]")
    # Decoder 需要接收它自己的输入(dec_inputs)以及 Encoder的输出(cross = enc_out)
    decoder.eval()
    with torch.no_grad():
        dec_out = decoder(dec_inputs, cross=enc_out)
    print(f"Decoder 输入维度 -> {dec_inputs.shape}")
    print(f"Encoder 提供的高级语义特征维度 (Cross) -> {enc_out.shape}")
    print(f"Decoder 最终投影输出维度 -> {dec_out.shape}  # [Batch, dec_in_len, 1]")
    
    print("="*50)
    print("✅ 测试通过：序列维度推演和数据流向正确！")