## from https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn import metrics
from ERR import cross_point
# import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
device = torch.device('cuda')



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

## 7. ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        ## 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x n_heads x len_k x d_v]
        ##首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        ## 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用 # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


## 6. MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V):

        ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        ##输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        ##下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以一看这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        ## 输入进行的attn_mask形状是 batch_size x len_q x len_k，然后经过下面这个代码得到 新的attn_mask : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个头上


        ##然后我们计算 ScaledDotProductAttention 这个函数，去7.看一下
        ## 得到的结果有两个：context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]


## 8. PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)




## 5. EncoderLayer ：包含两个部分，多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):
        ## 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


## 2. Encoder 部分包含三个部分：注意力层及后续的前馈神经网络

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) ## 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；

    def forward(self, enc_inputs):
        ## 这里我们的 enc_inputs 形状是： [batch_size，数据点长度，特征长度]

        enc_outputs = enc_inputs
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1)
        enc_self_attns = []
        for layer in self.layers:
            ## 去看EncoderLayer 层函数 5.
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns



## 1. 从整体网路结构来看，分为三个部分：编码层，解码层，输出层
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.linear= nn.Linear(3,d_model,bias=False)
        self.encoder = Encoder()  ## 编码层
        # 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 tgt_vocab_size 大小的softmax
        # self.projection = nn.Linear(d_model, 5, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(in_features=43904, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, enc_inputs):
        ## 输入大小为[32,3,686]
        enc_outputs=enc_inputs.transpose(-1,-2)
        ## 转换之后大小为[32,686,3]
        enc_outputs = self.linear(enc_outputs)
        ## 经过线性层大小为[32,686,64]
        enc_outputs, enc_self_attns = self.encoder(enc_outputs)
        ## 得到大小为[32,686,64]
        # enc_outputs = self.projection(enc_outputs)
        ## 大小为[32,768,5]
        enc_outputs=enc_outputs.view(-1,43904)
        ## 大小为[32,3840]
        enc_outputs =self.fc(enc_outputs)
        ## 大小为[32,5]
        return enc_outputs,enc_self_attns



if __name__ == '__main__':


    # 模型参数
    d_model = 64  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 1  # number of heads in Multi-Head Attention

    ##读取数据
    data = np.load('./res/sensor2data.npy')
    label = np.load('./res/sensor2lab.npy')
    X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.2)
    print(X_train.shape, X_test.shape)
    print(y_train, y_test)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    print(X_train.shape)
    print(X_test.shape)
    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

    Model = model().to(device)

    optimizer = torch.optim.Adam(Model.parameters(), lr=0.00001)
    loss_fun = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(20):
        Model.train()
        trainnum=0
        for i, (t, l) in enumerate(train_loader):
            out,att= Model(t)
            loss = loss_fun(out, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == 0:
                print('\n epoch: {}, i:{}, loss: {}'.format(epoch, i, loss))
            pred = torch.max(out, 1)[1]
            trainnum += torch.sum(pred == l)
        print('epoch: {}, acc: {}'.format(epoch, trainnum * 1.0 / 560))
        Model.eval()
        num = 0
        for i, (t, l) in enumerate(test_loader):
            out,att= Model(t)
            pred = torch.max(out, 1)[1]
            num += torch.sum(pred == l)
        print('epoch: {}, acc: {}'.format(epoch, num * 1.0 / 140))

    print('----------------')
    Model.eval()
    num = 0
    for i, (t, l) in enumerate(test_loader):
        out ,att= Model(t)
        # print(out)
        pred = torch.max(out, 1)[1]
        print(pred)
        print(l)
        num += torch.sum(pred == l)
    print('Test acc: {}'.format(num * 1.0 / 140))


