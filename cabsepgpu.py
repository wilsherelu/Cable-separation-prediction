import torch
import math
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import torch.nn.functional as F
import pandas as pd
import scipy.io as io
import collections
# torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 1      # rnn time step [s]

LR = 0.00005         # learning rate, for fresh training, the value should be much higher
train= False   #True   False
WD=1e-6

avgnum=1
if train:
    avgnum = 10
Npara=10
totaltime=1


dt=1/3600
exp=np.exp
totalstep=torch.tensor([int(totaltime*3600/TIME_STEP)])





class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.inexp = torch.nn.Linear(5, 1)
        self.k1 = torch.nn.Linear(7, 10)
        self.c1= torch.nn.Linear(7, 10)
        self.k2 =  torch.nn.Linear(10, 1)
        self.c2 =  torch.nn.Linear(10, 1)

    def forward(self,seprate,inputdata,ArrayTS,totaltime):

        imneg=self.inexp(torch.exp(inputdata))
        #inputdata[:,:,4] impact energy
        accimold=torch.log10(1/(inputdata[:,:,4].unsqueeze(2)*totaltime.unsqueeze(2)))


        INpos = torch.cat((seprate.unsqueeze(2), imneg,inputdata), dim=2)
        INneg=torch.cat((accimold,imneg, inputdata), dim=2)

        inputoutpos=self.k2(self.k1(INpos)).squeeze(2)
        inputoutneg = self.c2(self.c1(INneg)).squeeze(2)
        imdt=torch.sigmoid(inputoutpos)
        Reqim=torch.sigmoid(inputoutneg)

        Rsepdt = (imdt)*ArrayTS*(1-seprate)*(1-Reqim)

        newseprate = seprate +Rsepdt


        return newseprate

def weight_init(m): #initialization for fresh traning
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class My_loss(nn.Module): #customized loss function
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss=0.99*torch.mean(torch.pow((x - y), 2))+0.01*torch.mean(torch.pow((x - y)/(y), 2))
        return loss



rnn = RNN()

rnn.apply(weight_init)
rnn = rnn.cuda()

opt_rnn = torch.optim.Adam(rnn.parameters(), lr=LR,betas=(0.9, 0.999),weight_decay=WD)

loss_func=My_loss()
loss_func = loss_func.cuda()
Batchsize=6




lossrecord=torch.zeros(1).cuda()
Rt= torch.zeros(Batchsize,Npara)#separation yield t
Rt[:,:]=1E-4
Rt=Rt.cuda()
Rtnew= torch.zeros(Batchsize,Npara).cuda()#separation yield t+1
Expt= torch.zeros(Batchsize,Npara).cuda()#Experiment-based separation yield label


explength=torch.from_numpy(np.loadtxt('explength.csv',delimiter=',')).to(torch.float32).cuda() #Treatment time of each conditions
Expcond=torch.from_numpy(np.loadtxt('condition.csv',delimiter=',')).to(torch.float32).cuda() #Input features
timedata=torch.from_numpy(np.loadtxt('timedata.csv',delimiter=',')).to(torch.float32).cuda() #Experimental data
timedataend=torch.from_numpy(np.loadtxt('timedataend.csv',delimiter=',')).to(torch.float32).cuda() #Experimental data
expdata=torch.from_numpy(np.loadtxt('expdata.csv',delimiter=',')).to(torch.float32).cuda()#Experimental data

Expcond=Expcond.unsqueeze(1).repeat(1,Npara,1)
Expcond=F.normalize(Expcond,p=2.0, dim=0) #Input features normalize

avgpara=rnn.state_dict()
for param_tensor in avgpara:
    avgpara[param_tensor] = torch.zeros(avgpara[param_tensor].size()).cuda()

# rnn.load_state_dict(torch.load('trainedmodel.pth')) #for the second training or validation


torch.autograd.set_detect_anomaly(True)
#for data output into excel
outputdataX=torch.zeros(Batchsize,totalstep).cuda()
outputtime=torch.zeros(Batchsize,totalstep).cuda()
outputdataTrain=torch.zeros(Batchsize,totalstep,avgnum).cuda()
outputerror=torch.zeros(Batchsize,totalstep).cuda()




totalstep=totalstep.cuda()
#asynchronous timesteps
timestep1d=dt*torch.tensor([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1])
timestep1d=timestep1d.cuda()
timestep=torch.mm(explength.unsqueeze(1), timestep1d.unsqueeze(0))
for traintime in range(avgnum):
    Rt[:,:]=1E-4


    for step in range(1, totalstep):


        time = step * timestep
        #Stepwise function
        for i in range(Npara):

            for j in range(time.size(0)):
                Rindex = torch.nonzero((time[j, i] < timedataend[j, :]) & (time[j, i] > timedata[j, :]), as_tuple=False)

                Expt[j, i] = expdata[j, Rindex] + (expdata[j, Rindex + 1] - expdata[j, Rindex]) \
                             / (timedata[j, Rindex + 1] - timedata[j, Rindex]) * (time[j, i] - timedata[j, Rindex])

        if train:
            for lt in range(5):


                Rtnew = rnn(Rt, Expcond, timestep,  time)

                lossrnn = loss_func(Rtnew, Expt)
                opt_rnn.zero_grad()  # clear gradients for this training step
                lossrnn.backward(retain_graph=True)  # backpropagation, compute gradients
                opt_rnn.step()  # apply gradients

            outputdataTrain[:,step,traintime]=Rtnew.data[:,0]


        Rtnew = rnn(Rt, Expcond, timestep,  time)

        Rt = Rtnew.data

        opt_rnn.zero_grad()


        if not(train):
            outputdataX[:,step]=Rt[:,0]
            outputtime[:,step]=time[:,0]


    plt.close()
    #save mean parameters based on the average considering loss
    if train:
        lossrecord=lossrecord+1/lossrnn.data
        for param_tensor in avgpara:
            avgpara[param_tensor] = avgpara[param_tensor]+rnn.state_dict()[param_tensor]/lossrnn.data
    collections.defaultdict(dict)

for param_tensor in avgpara:
    avgpara[param_tensor] = avgpara[param_tensor]/lossrecord
#save model parameters  or predicted data
if train:
    path = 'trainedmodel.pth'
    torch.save(avgpara, path)
    outputdataX=torch.mean(outputdataTrain,dim=2)
    outputerror=torch.std(outputdataTrain,dim=2)
    savedata1 = np.array(outputdataX.cpu().detach().numpy())
    outdata1 = pd.DataFrame(savedata1)
    writer = pd.ExcelWriter('traindata.xlsx')  #
    outdata1.to_excel(writer, 'mean', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    savedata2 = np.array(outputerror.cpu().detach().numpy())
    outdata2 = pd.DataFrame(savedata2)
    outdata2.to_excel(writer, 'std', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.close()

# if not(train):
#     savedata1 = np.array(outputdataX.cpu().detach().numpy())
#     outdata1 = pd.DataFrame(savedata1)
#     writer = pd.ExcelWriter('validate.xlsx')
#     outdata1.to_excel(writer, 'X', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
#     savedata2 = np.array(outputtime.cpu().detach().numpy())
#     outdata2 = pd.DataFrame(savedata2)
#     outdata2.to_excel(writer, 'TIME', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
#     writer.close()
# io.savemat('savedia.mat',{'savedata':savedata1})
# io.savemat('saveX.mat',{'savedata':savedata2})
