import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
from ERR import cross_point
device = torch.device('cuda')


class model(nn.Module):
    def __init__(self, num_classes):

        super(model, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=10, stride=10)  # padding也可以是tuple:(1,2)，#【输入信号的通道，卷积产生的通道(卷积核个数)，卷积核尺寸，卷积步长】，输入：【batch_size，信号维度，信号的最大长度】
        # self.conv2 = nn.Conv1d(64,128, kernel_size=5, stride=1)  # padding也可以是tuple:(1,2)，
        # self.conv3 = nn.Conv1d(128,128, kernel_size=5, stride=1)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1,batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1,batch_first=True ,bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = F.relu(self.conv1(x)) #输入：【batch_size，信号维度，信号的最大长度】
        # print("*****************")
        # print(out.shape)
        # out = F.relu(self.conv2(out))
        # print("*****************")
        # print(out.shape)
        # out = F.relu(self.conv3(out))
        # print("*****************")
        # print(out.shape)
        out = F.dropout(out, p=0.5)
        # print("*****************")
        # print(out.shape)
        #out=out.permute(2,0,1)
        out=out.transpose(2,1)
        batch_size = x.shape[0]
        hidden_state = torch.randn(2 , batch_size, 128).to(device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(2, batch_size, 128).to(device)
        out, (hn, cn) = self.lstm1(out, (hidden_state,cell_state))
        # #print("*****************")
        # print(out.shape)
        # out = F.dropout(out, p=0.5)
        # print("*****************")
        # print(out.shape)
        out, (hn, cn) = self.lstm2(out, (hn,cn))
        # print(out.shape)
        # output_last = out[:, 0, -128:]
        # # 获反向最后一层的h_n
        # h_n_last = hn[1]
        #
        # print(output_last.size())
        # print(h_n_last.size())
        # # 反向最后的output等于最后一层的h_n
        # print(output_last.eq(h_n_last))
        #
        # # 获取正向的最后一个output
        # output_last = out[:, -1, :128]
        # # 获取正向最后一层的h_n
        # h_n_last = hn[0]
        # # 反向最后的output等于最后一层的h_n
        # print(output_last.eq(h_n_last))
        # output=hn
        # print(output.shape)
        # output = output.transpose(1, 0)
        # print(output.shape)
        # output = torch.reshape(output, (-1, 256))
        # print(output.shape)
        # print("*****************")
        # print(out.shape)
        # output = F.dropout(out, p=0.5)
        out = F.dropout(out, p=0.5)
        # print("*****************")
        # print(out[:,-1,:].shape)
        # out = self.fc(output)
        # out = torch.reshape(out, (-1, 288*64))
        out = self.fc(out[:,-1,:])  #  (batch_size, num_layers, hidden_size)
        # print(out.shape)
        # out = self.fc(out.reshape(out.shape[0], 25 * 64))
        return out


X_train, X_test, y_train, y_test = np.load('./res/train1.npy'), np.load('./res/test1.npy'), np.load(
    './res/train_lab1.npy'), np.load('./res/test_lab1.npy')
print(X_train.shape, X_test.shape)
print(y_train, y_test)
# 原始lable是1,2，需要改为0,1
# y_train = y_train - 1
# y_test = y_test - 1
# print(y_train)
print(sum(y_test==1))
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
print(X_train.shape[0])
print(X_test.shape[0])
train_set = TensorDataset(X_train, y_train)
test_set = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_set, batch_size=64,shuffle=True)
test_loader = DataLoader(test_set, batch_size=64,shuffle=True)

Model = model(2).to(device)

optimizer = torch.optim.Adam(Model.parameters(), lr=0.00001)
loss_fun = torch.nn.CrossEntropyLoss().to(device)
losslist=[]
acclist=[]
for epoch in range(200):
    Model.train()
    for i, (t, l) in enumerate(train_loader):
        out = Model(t)
        loss = loss_fun(out, l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i == 0:
            print('\n epoch: {}, i:{}, loss: {}'.format(epoch, i, loss))
            losslist.append(loss.cpu().detach().numpy())

    Model.eval()
    num = 0
    for i, (t, l) in enumerate(test_loader):
        out = Model(t)
        pred = torch.max(out, 1)[1]
        num += torch.sum(pred == l)
    print('epoch: {}, acc: {}'.format(epoch, num * 1.0 / 1180))
    acclist.append((num * 1.0 / 1180).cpu().detach().numpy())

print('----------------')
Model.eval()
num = 0
far = 0
frr = 0
test_y = []
prediction_y = []
test_yf = []
pred_yf = []
for i, (t, l) in enumerate(test_loader):
    out = Model(t)
    # print(out)
    pred = torch.max(out, 1)[1]
    pred_y = out.cpu().detach().numpy()
    test_label = l.cpu().numpy()
    prediction_y.append(pred_y)
    test_y.append(test_label)
    test_yf.append(l.cpu().detach().numpy())
    pred_yf.append(pred.cpu().detach().numpy())
    # print(pred)
    pred_frr=(pred==0).nonzero()
    # print(pred_frr)
    pred_far=(pred==1).nonzero()
    label_frr = (l == 0).nonzero()
    # print(label_frr)
    label_far = (l == 1).nonzero()
    num += torch.sum(pred == l)
    for n in pred_frr:
        if torch.sum(label_frr == n):
            frr=frr+1
    for n in pred_far:
        if torch.sum(label_far == n):
            far=far+1
print('Test acc: {}'.format(num * 1.0 / 1180))
print(num,frr,far)
# torch.save(Model,'model_3.pth')
print('FRR:{}'.format(1-(frr*1.0/800)))
print('FAR:{}'.format(1-(far*1.0/380)))
y_true = np.concatenate(test_yf)
y_pred = np.concatenate(pred_yf)
precision, recall, f1 = precision_recall_fscore_support(y_true,y_pred,average='binary',pos_label=0)[:-1]
print('precision:{}'.format(precision))
print('recall:{}'.format(recall))
print('f1:{}'.format(f1))
# np.save('./res/cnnLSTMp.npy',prediction_y)
# np.save('./res/cnnLSTMl.npy',test_y)
# np.save('./res/cnnBiLSTMloss1.npy',losslist)
# np.save('./res/cnnBiLSTMacc1.npy',acclist)
prediction_y = np.concatenate(prediction_y)
test_y = np.concatenate(test_y)
fpr,tpr,thresholds = metrics.roc_curve(test_y,prediction_y[:,0],pos_label=0)
roc_auc = metrics.auc(fpr,tpr)
index=np.argmin(np.abs(1-tpr-fpr))
frr=(1-tpr)
if(frr[index]>fpr[index]):
    if frr[index+1]<fpr[index+1]:
        line1=[1,fpr[index],2,fpr[index+1]]
        line2=[1,frr[index],2,frr[index+1]]
        print("EER:{}".format(cross_point(line1,line2)[1]))
elif frr[index]==fpr[index]:
    print("EER:{}".format(frr[index]))
else:
    if frr[index-1]>fpr[index-1]:
        line1=[1,fpr[index-1],2,fpr[index]]
        line2=[1,frr[index-1],2,frr[index]]
        print("EER:{}".format(cross_point(line1,line2)[1]))


