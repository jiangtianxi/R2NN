import torch
import matplotlib.pyplot as plt
import numpy as np
import metarnn
from scipy import io
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import datetime
from metarnn import generateCoupMat

## Step1:加载数据
class locationDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

time_now = datetime.datetime.now ().strftime ('%Y_%m_%d_%H_%M')
train_log_filename = "train_log" + time_now + ".txt"
train_log_filepath = "./log/"+ train_log_filename
train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [proportion] {prop} [Lr*1e-5] {lr} [Loss] {loss_str}\n"

# 读取数据
directionary = "./data"
filename = "sample"
num_class = 3
num_sample = 1000  # 每一个类别的样本数
len_sample = 2000  # 每一个样本的长度
x = np.zeros((1, len_sample))
for i in range(num_class):
    dataFile = directionary + '/' + filename + str(i + 1) + '.mat'
    data = np.transpose(io.loadmat(dataFile)['Sample'])
    x = np.vstack((x, data))
x = np.delete(x, 0, 0)
y = np.zeros((1, 1))
for i in range(num_class):
    label = i * np.ones((num_sample, 1))
    y = np.vstack((y, label))
y = np.delete(y, 0, 0)
x = torch.from_numpy(x)
y = torch.from_numpy(y).reshape(num_sample * num_class, )

fullDataset = locationDataset(x, y)
train_size  = int(0.8 * len(fullDataset))
test_size   = len(fullDataset) - train_size
dataset_train, dataset_test = torch.utils.data.random_split(
    fullDataset, 
    [train_size, test_size], 
    generator=torch.Generator().manual_seed(0)
)

batch_size = 240

train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                           batch_size=600,
                                           shuffle=False)
## Step2:定义RNN模型参数
fs = 1e3
beta = 1
dt = 1
row, col = 5, 5  #(9,9)
dim = 2 * row * col
domain_shape = (dim, 1)

# history value:
m_c = 13.033e-3 * torch.ones(1, row * col)
m_n = 35.258e-3 * torch.ones(1, row * col)

torch.manual_seed(0)
# k_n = 1e-3 * torch.rand(1, row * col, dtype=torch.float32) / beta**2
# k_c = 1e-3 * torch.rand(1, (row - 1) * col + row * (col - 1), dtype=torch.float32) / beta**2

# history value: 0.2e-3; 0.5578e-3;
# k_n = 3.186e-3 * torch.ones(1, row * col) / beta**2
k_n = 1.0e-3 + 6.0e-3 * torch.rand(1, row * col, dtype=torch.float32) / beta**2

k_c = 5e-3 * torch.ones(1, (row - 1) * col + row * (col - 1)) / beta**2
# k_c = 0.2e-3 + 0.8e-3 * torch.rand(1, (row - 1) * col + row * (col - 1), dtype=torch.float32) / beta**2
# history value: k_c = 0.2e-3 + 0.8e-3 * torch.rand(1, (row - 1) * col + row * (col - 1), dtype=torch.float32) / beta**2

c_n = 0.0 * torch.ones(1, row * col) / beta
c_c = 0.0 * torch.ones(1, (row - 1) * col + row * (col - 1)) / beta

kn_o = 1e-3 * torch.ones(1, row * col) / beta**2
kc_o = 1e-3 * torch.rand(1, (row - 1) * col + row * (col - 1), dtype=torch.float32) / beta**2

coupMat = generateCoupMat.init_coupling_mat(row, col, k_c, k_n, m_c, m_n, c_c, c_n)

coup = metarnn.Coupling(domain_shape, row, col, m_n, m_c, k_n, k_c, c_n, c_c, coupMat, kn_o, kc_o)
cell = metarnn.WaveCell(dt, coup)
srcpos = 2  #(4)
src = metarnn.WaveSource(srcpos, dt)
# (scrpos:prb)
fxp = [0, 4, 20, 24]  #[0, 8, 72, 80]
#(36,44,76)
prb1 = 35
prb2 = 39
prb3 = 47
probe = [metarnn.WaveIntensityProbe(prb1),
         metarnn.WaveIntensityProbe(prb2),
         metarnn.WaveIntensityProbe(prb3)]

## Step3: 训练模型
model = metarnn.WaveRNN(cell, src, dt, fxp, srcpos, probe)
Lr = 1e-4  # init learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=Lr)
loss_iter = []
time_pass = []
lr_list = []

for param in model.parameters():
    print(type(param.data), param.size())

def train(epoch, max_epoch, model_filename):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        
        X = data.unsqueeze(2)  # dimension: (batch_size, time, 1)
        output = metarnn.utils.normalize_power(model(X).sum(dim=1))
                    
        criterion = torch.nn.CrossEntropyLoss()
        # loss
        print(target.long().size())
        loss = criterion(output, target.long())
        optimizer.zero_grad()
        loss.backward()
        # update
        optimizer.step()

        # reduce learning rate
        for p in optimizer.param_groups:
            p['lr'] *= 0.992
#             p['lr'] *= 1
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

        print("Epoch: {} -- [{}/{} ({}%)] -- Lr: {:.3f} -- Loss: {:.5f}".format(
            epoch, (batch_idx + 1), len(train_loader),
            100. * (batch_idx + 1) / len(train_loader),
            lr_list[-1]*1e5, loss.item()))
        to_write = train_log_txt_formatter.format(time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
                                                  epoch = epoch,
                                                  prop = "{}/{}".format((batch_idx + 1), len(train_loader)),
                                                  lr = "{:.4f}".format(lr_list[-1]*1e5),
                                                  loss_str = "{:.5f}".format(loss))
        with open(train_log_filepath,"a") as f:
            f.write(to_write)
        loss_iter.append(loss.item())
        
        if batch_idx + 1 == len(train_loader):
            torch.save(model, "./model/" + model_filename)
            print('\n')
        
    return loss.item()

print('training has already started ...')

def test(net):
    test_loss = 0
    correct = 0
    # 测试集
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)

        X = data.unsqueeze(2)
        u = net.forward(X, output_fields=True, if_test=True)
        output = metarnn.utils.normalize_power(u[:, :, [prb1, prb2, prb3], 0].pow(2).sum(dim=1))

        # sum up batch loss
        # test_loss += F.nll_loss(output, target.long()).item()
        criterion = torch.nn.CrossEntropyLoss()
        test_loss += criterion(output, target.long()).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    to_write = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    with open(train_log_filepath, "a") as f:
        f.write(to_write)
    return pred, output, u

max_epoch = 30
model_filename_list = []
for epoch in range(1, max_epoch+1):
    model_filename = "model" + time_now + "_epoch_" + str(epoch) + ".pkl"
    model_filename_list.append(model_filename)
    starttime = datetime.datetime.now()
    val_loss = train(epoch, max_epoch, model_filename)  
    endtime = datetime.datetime.now()
    time_pass.append(endtime - starttime) 
    
print('training has already finished ...\n')

# Step4: 测试模型推理准确率与epoch关系
for epoch in range(0, max_epoch):
    net = torch.load("./model/" + model_filename_list[epoch])
    
    print('testing has already started ...\nEpoch: ' + str(epoch))
    [pred, output, u] = test(net)
    print('testing has already finished ...' )
    
    kc = net.cell.coup.kc
    kn = net.cell.coup.kn
    np.savetxt('parameter_save/parameter_kc' + time_now + "_epoch_" + str(epoch) + '.txt', kc.detach().numpy())
    np.savetxt('parameter_save/parameter_kn' + time_now + "_epoch_" + str(epoch) + '.txt', kn.detach().numpy())
