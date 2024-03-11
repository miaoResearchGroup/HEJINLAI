## from https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn import metrics
from ERR import cross_point
from torchsummary import summary
# import matplotlib.pyplot as plt
import math
import time
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
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    def forward(self, enc_inputs):
        enc_outputs = enc_inputs
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1)
        print(enc_outputs.shape)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            print(enc_self_attn.shape)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns



## 1. 从整体网路结构来看，分为三个部分：编码层，解码层，输出层
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.linear= nn.Linear(3,d_model,bias=False)
        self.encoder = Encoder()
        self.projection = nn.Linear(d_model, 2, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(in_features=19200, out_features=2),
            nn.Softmax(dim=1)
        )
    def forward(self, enc_inputs):
        enc_outputs=enc_inputs.transpose(-1,-2)
        enc_outputs = self.linear(enc_outputs)
        # print(enc_outputs.shape)
        enc_outputs, enc_self_attns = self.encoder(enc_outputs)
        enc_outputs=enc_outputs.view(-1,19200)
        enc_outputs =self.fc(enc_outputs)
        return enc_outputs,enc_self_attns



if __name__ == '__main__':


    # 模型参数
    d_model = 64  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 2  # number of heads in Multi-Head Attention

    ##读取数据
    X_train, X_test, y_train, y_test = np.load('./res/train1.npy'), np.load('./res/test1.npy'), np.load(
        './res/train_lab1.npy'), np.load('./res/test_lab1.npy')
    print(X_train.shape, X_test.shape)
    print(y_train, y_test)
    # 原始lable是1,2，需要改为0,1
    # y_train = y_train - 1
    # y_test = y_test - 1
    # print(y_train)
    print(sum(y_test == 1))
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
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    Model = model().to(device)

    optimizer = torch.optim.Adam(Model.parameters(), lr=0.00001)
    loss_fun = torch.nn.CrossEntropyLoss().to(device)

    losslist = []
    acclist = []
    start_time = time.time()
    # summary(Model,(3,300))
    for epoch in range(100):
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
                losslist.append(loss.cpu().detach().numpy())
            pred = torch.max(out, 1)[1]
            trainnum += torch.sum(pred == l)
        print('epoch: {}, acc: {}'.format(epoch, trainnum * 1.0 / 780))
        Model.eval()
        num = 0
        for i, (t, l) in enumerate(test_loader):
            out,att= Model(t)
            pred = torch.max(out, 1)[1]
            num += torch.sum(pred == l)
        print('epoch: {}, acc: {}'.format(epoch, num * 1.0 / 1180))
        acclist.append((num * 1.0 / 1180).cpu().detach().numpy())

    print('----------------')
    end_time = time.time()
    # 计算程序运行时间
    run_time = end_time - start_time

    # 输出运行时间
    print("程序运行时间：", run_time, "秒")
    Model.eval()
    num = 0
    far = 0
    frr = 0
    test_y = []
    prediction_y = []
    test_yf = []
    pred_yf = []
    att_y = []
    data_y = []
    for i, (t, l) in enumerate(test_loader):
        out ,att= Model(t)
        # print(out)
        pred = torch.max(out, 1)[1]
        pred_y = out.cpu().detach().numpy()
        for j in range(6):
            att_y.append(att[j].cpu().detach().numpy())
        # att_y.append(att[5].cpu().detach().numpy())
        data_y.append(t.cpu().detach().numpy())
        test_label = l.cpu().numpy()
        prediction_y.append(pred_y)
        test_y.append(test_label)
        test_yf.append(l.cpu().detach().numpy())
        pred_yf.append(pred.cpu().detach().numpy())
        # print(pred)
        pred_frr = (pred == 0).nonzero()
        # print(pred_frr)
        pred_far = (pred == 1).nonzero()
        label_frr = (l == 0).nonzero()
        # print(label_frr)
        label_far = (l == 1).nonzero()
        num += torch.sum(pred == l)
        # 取出正确的个数
        for n in pred_frr:
            if torch.sum(label_frr == n):
                frr = frr + 1
        for n in pred_far:
            if torch.sum(label_far == n):
                far = far + 1
        # frr += torch.sum(pred_frr == label_frr)
        # far += torch.sum(pred_far == label_far)
    print('Test acc: {}'.format(num * 1.0 / 1180))
    # print(frr)
    # print(far)
    print('FRR:{}'.format(1 - (frr * 1.0 / 800)))

    print('FAR:{}'.format(1 - (far * 1.0 / 380)))
    y_true = np.concatenate(test_yf)
    y_pred = np.concatenate(pred_yf)
    precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=0)[:-1]
    print('precision:{}'.format(precision))
    print('recall:{}'.format(recall))
    print('f1:{}'.format(f1))
    att_y = np.array(att_y)
    # np.save('./res/t.npy',data_y)
    # np.save('./res/l.npy',test_y)
    # np.save('./res/loss.npy',losslist)
    # np.save('./res/acc.npy',acclist)
    # np.save('./res/att.npy',att_y)
    # torch.save(Model,'model_5.pth')
    prediction_y = np.concatenate(prediction_y)
    test_y = np.concatenate(test_y)
    fpr, tpr, thresholds = metrics.roc_curve(test_y, prediction_y[:, 0], pos_label=0)
    roc_auc = metrics.auc(fpr, tpr)
    index = np.argmin(np.abs(1 - tpr - fpr))
    frr = (1 - tpr)
    if (frr[index] > fpr[index]):
        if frr[index + 1] < fpr[index + 1]:
            line1 = [1, fpr[index], 2, fpr[index + 1]]
            line2 = [1, frr[index], 2, frr[index + 1]]
            print("EER:{}".format(cross_point(line1, line2)[1]))
    elif frr[index] == fpr[index]:
        print("EER:{}".format(frr[index]))
    else:
        if frr[index - 1] > fpr[index - 1]:
            line1 = [1, fpr[index - 1], 2, fpr[index]]
            line2 = [1, frr[index - 1], 2, frr[index]]
            print("EER:{}".format(cross_point(line1, line2)[1]))



