import config
import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = config.device

def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 词嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        '''
        Args:
            d_model: 传入模型的维度
            vocab: 词汇表的大小
        Returns: /
        '''
        super(Embeddings, self).__init__()
        # 将词汇表映射为d_model维的向量
        self.lut = nn.Embedding(vocab, d_model)
        # 储存模型的维度
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        # 这是为了保持词向量的方差，使其适应后续层的训练。
        return self.lut(x) * math.sqrt(self.d_model)

# 位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        '''
        Args:
            d_model: 传入模型的维度
            dropout: dropout率
            max_len: 最大长度
        Returns: /
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个 max_len * embedding 维度大小的全零矩阵
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        # 计算位置向量
        div_term = torch.exp(torch.arange(0., d_model,2, device=DEVICE) * -(math.log(10000.0) / d_model))

        # 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 加1个维度，使得pe维度变为：1×max_len×embedding维度
        # (方便后续与一个batch的句子所有词的embedding批量相加)
        pe = pe.unsqueeze(0)
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# 注意力机制核心代码
def attention(query, key, value, mask=None, dropout=None):
    '''
    Args: 计算query与key之间的点积
        query: 输入query矩阵
        key: 输入key矩阵
        value: 输入value矩阵
        mask: 是否使用掩码
        dropout: 是否使用dropout
    Returns:
        torch.matmul(p_attn, value): 注意力与v的点积结果
        p_attn: 注意力得分矩阵
    '''
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)

    # 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    # scores为注意力机制得分矩阵
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)

    # 将mask后的attention矩阵按照最后一个维度进行softmax
    p_attn = F.softmax(scores, dim=-1)

    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn

# 多头注意力机制
class MultiHeadedAttention(nn.Module):
    '''
    原始单头注意力：
        [batch_size, seq_len, d_model]
        → 注意力计算 →
        [batch_size, seq_len, d_model]
    多头注意力：
        [batch_size, seq_len, d_model]
        → 分成h个头 [batch_size, h, seq_len, d_k]
        → 每个头独立计算注意力
        → 合并结果 [batch_size, seq_len, d_model]
    '''
    def __init__(self, h, d_model, dropout=0.1):
        '''
        Args:
            h: 注意力机制头的数量
            d_model: 模型的嵌入维度
            dropout: dropout率
        '''
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0
        # 得到一个head的attention表示维度
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query的第一个维度值为batch size
        nbatches = query.size(0)
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        '''
        1. 线性变换
            l(x)  # 对query, key, value分别进行线性变换
            使用不同的全连接层将输入映射到新的空间。
        2. 形状重塑和维度变换
            假设输入形状：[batch_size, seq_len, d_model]
            .view(nbatches, -1, self.h, self.d_k)
            将形状变为：[batch_size, seq_len, h, d_k]
            .transpose(1, 2)
            将形状变为：[batch_size, h, seq_len, d_k]
        完整流程示例
            假设：
                batch_size = 2
                seq_len = 10
                d_model = 512
                h = 8 (8个注意力头)
                d_k = 64 (512 ÷ 8 = 64)
            变换过程：
                输入: query.shape = [2, 10, 512]
                线性变换后: [2, 10, 512] (保持不变)
                view重塑: [2, 10, 8, 64] (分成8个头，每个头64维)
                transpose转置: [2, 8, 10, 64] (现在每个头独立处理)
        '''
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.alpha = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 输入x是神经网络层的输出
        # 按最后一个维度计算均值和方差
        # keepdim=True确保输出的维度与输入相同
        mean = x.mean(-1, keepdim=True)  # 计算最后一个维度的均值
        std = x.std(-1, keepdim=True)  # 计算最后一个维度的标准差

        # 返回Layer Norm的结果
        # Layer Norm公式: y = a * (x - mean) / sqrt(std^2 + eps) + b
        # 其中a和b是可学习的参数，eps是为了防止除以0的小常数
        return self.alpha * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.beta


# 前馈神经网络:两个线性层+ReLU
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        位置前馈神经网络初始化函数
        参数:
            d_model: 模型的输入维度
            d_ff: 前馈神经网络中间层的维度
            dropout: dropout概率，默认为0.1
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

# 将Multi-Head Attention和Feed Forward层连在一起
class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层
    连在一起只不过每一层输出之后都要先做Layer Norm再残差连接
    sublayer是lambda函数
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    """
    sublayer参数是SublayerConnection类中的核心组件，
    它代表了Transformer中的具体处理层，在这里使用的可以是多头注意力块、前馈神经网络块等
    """
    def forward(self, x, sublayer):
        # 返回Layer Norm和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))

# 编码器实现
# 实现单个编码器
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        '''
        Args:
            size: 归一化层的大小
            self_attn: 使用的多头注意力机制函数
            feed_forward: 使用的前向传播函数
            dropout: dropout率
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # SublayerConnection的作用就是把multi和ffn连在一起
        self.sublayer1 = SublayerConnection(size, dropout)  # 第一个子层（自注意力）
        self.sublayer2 = SublayerConnection(size, dropout)  # 第二个子层（前馈网络）
        # d_model
        self.size = size

    def forward(self, x, mask):
        # 第一个子层：多头自注意力
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        # 第二个子层：前馈网络
        x = self.sublayer2(x, self.feed_forward)
        return x

# N个编码器组成编码器整体模块
class Encoder(nn.Module):
    # N = 6 论文中
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        使用循环连续eecode N次(这里为6次)
        这里的Eecoderlayer会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# 解码器实现
# 单个解码器
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer1 = SublayerConnection(size, dropout)  # 第一个子层（自注意力）
        self.sublayer2 = SublayerConnection(size, dropout)  # 第二个子层（交叉注意力）
        self.sublayer3 = SublayerConnection(size, dropout)  # 第三个子层（前馈网络）

    def forward(self, x, memory, src_mask, tgt_mask):  # src_mask：填充掩码  tgt_mask：因果掩码
        # 用m来存放encoder的最终hidden表示结果
        m = memory
        # Self-Attention：注意self-attention的q，k和v均为decoder hidden
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.sublayer2(x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer3(x, self.feed_forward)

# N个解码器组成解码器整体模块
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# 生成器部分
class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)

# 最终组合模型
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        '''
        Args:
            encoder: 编码器
            decoder: 解码器
            src_embed: 原始词表
            tgt_embed: 目标词表
            generator: 生成器
        '''
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    def forward(self, src, tgt, src_mask, tgt_mask):
        '''
        Args:
            src: 预测数据输入
            tgt: 已经预测出的目标
            src_mask: 是否使用填充掩码
            tgt_mask: 是否使用因果掩码
        Returns: 预测结果
        '''
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    '''
    Args:
        src_vocab: 原始词汇表
        tgt_vocab: 目标词汇表
        N: 编码器和解码器的个数
        d_model: 词汇表维度
        d_ff: 前馈神经网络隐藏层参数
        h: 多头注意力机制的头数
        dropout: drop率
    Returns: 模型训练结果
    '''
    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    # 实例化Transformer模型对象

    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),  # 编码器部分参数
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),  # 解码器部分参数
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)

    # 初始化模型参数
    # 遍历模型中的所有参数
    for p in model.parameters():
        # 判断参数是否为二维或更高维（例如权重矩阵，而不是偏置向量）
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)