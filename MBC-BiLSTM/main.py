import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
from ERR import cross_point
import time
device = torch.device('cuda')


class model(nn.Module):
    def __init__(self, num_classes):

        super(model, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=5, stride=5)  # padding也可以是tuple:(1,2)，#【输入信号的通道，卷积产生的通道(卷积核个数)，卷积核尺寸，卷积步长】，输入：【batch_size，信号维度，信号的最大长度】
        # self.conv2 = nn.Conv1d(64,64, kernel_size=5, stride=1)  # padding也可以是tuple:(1,2)，
        # self.conv3 = nn.Conv1d(64,64, kernel_size=5, stride=1)
        self.conv4 = nn.Conv1d(3, 64, kernel_size=20, stride=20)
        # self.conv5 = nn.Conv1d(64, 64, kernel_size=20, stride=1)
        # self.conv6 = nn.Conv1d(64, 64, kernel_size=20, stride=1)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1,batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1,batch_first=True ,bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1,batch_first=True ,bidirectional=True)
        self.lstm4 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1,batch_first=True ,bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out1 = F.relu(self.conv1(x)) #输入：【batch_size，信号维度，信号的最大长度】
        # out1 = F.relu(self.conv2(out1))
        # out1 = F.relu(self.conv3(out1))
        out1 = F.dropout(out1, p=0.5)
        out1=out1.transpose(2,1)
        batch_size = x.shape[0]
        hidden_state = torch.randn(2, batch_size, 128).to(device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(2, batch_size, 128).to(device)
        out1, (hn, cn) = self.lstm1(out1, (hidden_state,cell_state))
        out1, (hn, cn) = self.lstm2(out1, (hn,cn))
        output1=hn
        output1 = output1.transpose(1, 0)
        output1 = torch.reshape(output1, (-1, 256))
        output1 = F.dropout(output1, p=0.5)

        out2 = F.relu(self.conv4(x)) #输入：【batch_size，信号维度，信号的最大长度】
        # out2 = F.relu(self.conv5(out2))
        # out2 = F.relu(self.conv6(out2))
        out2 = F.dropout(out2, p=0.5)
        out2=out2.transpose(2,1)
        batch_size = x.shape[0]
        hidden_state = torch.randn(2, batch_size, 128).to(device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(2, batch_size, 128).to(device)
        out2, (hn, cn) = self.lstm1(out2, (hidden_state,cell_state))
        out2, (hn, cn) = self.lstm2(out2, (hn,cn))
        output2=hn
        output2 = output2.transpose(1, 0)
        output2 = torch.reshape(output2, (-1, 256))
        output2 = F.dropout(output2, p=0.5)
        output = torch.cat((output1,output2),dim=1)
        out = self.fc(output)
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
print(X_train.shape)
print(X_test.shape)
train_set = TensorDataset(X_train, y_train)
test_set = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_set, batch_size=64,shuffle=True)
test_loader = DataLoader(test_set, batch_size=64,shuffle=True)

Model = model(2).to(device)

optimizer = torch.optim.Adam(Model.parameters(), lr=0.00001)
loss_fun = torch.nn.CrossEntropyLoss().to(device)
losslist=[]
acclist=[]
start_time = time.time()
for epoch in range(100):
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
    acclist.append((num*1.0/1180).cpu().detach().numpy())

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
    # 取出正确的个数
    for n in pred_frr:
        if torch.sum(label_frr == n):
            frr=frr+1
    for n in pred_far:
        if torch.sum(label_far == n):
            far=far+1
    # frr += torch.sum(pred_frr == label_frr)
    # far += torch.sum(pred_far == label_far)
print('Test acc: {}'.format(num * 1.0 / 1180))
print('FRR:{}'.format(1-(frr*1.0/800)))
print('FAR:{}'.format(1-(far*1.0/380)))
y_true = np.concatenate(test_yf)
y_pred = np.concatenate(pred_yf)
precision, recall, f1 = precision_recall_fscore_support(y_true,y_pred,average='binary',pos_label=0)[:-1]
print('precision:{}'.format(precision))
print('recall:{}'.format(recall))
print('f1:{}'.format(f1))
# np.save('./res/MBcnnbiLSTMp.npy',prediction_y)
# np.save('./res/MBcnnbiLSTMl.npy',test_y)
np.save('./MBcnnbiLSTMloss.npy',losslist)
np.save('./MBcnnbiLSTMacc.npy',acclist)
# torch.save(Model,'model_MB.pth')
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




