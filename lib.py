import numpy as np
import os
import pylab
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset
import torch

class ParseGRU():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='', help='dataset directory')
        parser.add_argument('--log_folder', default='./logs', help='log directory')
        parser.add_argument('--batch_size', type=int,default=16)
        parser.add_argument('--video_batch', type=int,default=16)
        parser.add_argument('--image_size', default=64)
        parser.add_argument('--T', type=int, default=16, help='time steps')
        parser.add_argument('--check_point', type=int, default=10)
        parser.add_argument('--n_channels', type=int, default=1, help='number of channels') 
        parser.add_argument('--num_epochs', type=int, default=100 )
        parser.add_argument('--z_dim', type=int, default=64)
        parser.add_argument('--ngru', type=int, default=100)
        parser.add_argument('--learning_rate', type=int, default=1e-4)
        parser.add_argument('--cuda', type=bool, default=True)

        self.args = parser.parse_args()

class Visualizer():
    def __init__(self,opt):
        self.opt = opt

    def plot_loss(self):
        pylab.xlim(0, self.opt.num_epochs)  # *self.len
        pylab.ylim(0, max(self.losses))
        plt.plot(self.losses, label='loss')#'+','.join(self.opt.dis_loss))#if wanna print type of loss
        plt.legend()
        plt.savefig(os.path.join(self.opt.log_folder, 'loss.pdf'))
        plt.close()

class Seq_Dataset(Dataset):
    def __init__(self, datasets ,time_steps,slide=True,shuffle=True):
        self.dataset = datasets
        self.slide = slide
        self.shuffle = shuffle
        label = np.zeros(len(datasets))#sum of number of img file

        for i in range(len(datasets)):
            _,label[i] = datasets[i]#label = img belong which dir
        lener = np.zeros(int(max(label))+1)

        for i in range (int(max(label))+1):
            lener[i] = int(len(label[label==i]))#.sum()#label has about 3000 sha
        leners2 = np.array([])

        for i in range (int(max(label))+1):
            if i ==0:
                leners2 = np.append(leners2,lener[i])
            else:
                leners2 = np.append(leners2,lener[i]+leners2[i-1])
        ban_list = np.array([])

        for i in range(int(max(label))+1):
            if lener[i] == time_steps:
                ban = np.arange(leners2[i]+1 - time_steps, leners2[i])

            else:
                ban = np.arange(leners2[i]-time_steps,leners2[i])
            ban_list = np.append(ban_list,ban)
        vec = np.arange(int(max(leners2)))
        ind = [i for i in range(int(max(leners2))) if i not in ban_list]

        ok_list = vec[ind]
        self.leners2 = leners2
        self.time_steps = time_steps
        self.channels = self.dataset[0][0].size(0)
        self.img_size = self.dataset[0][0].size(1)
        video = torch.Tensor()
        self.video = video.new_zeros((time_steps,self.channels,self.img_size,self.img_size))
        self.indexes = np.array([])
        self.ban_list =  ban_list
        self.ok_list = ok_list

    def __len__(self):
        return int(max(self.leners2))#2870#int(np.sum(self.lener - self.time_steps))#len(self.dataset)-self.time_steps

    def __getitem__(self, index):
        if index in self.ban_list:

            if self.shuffle:
                index = np.random.choice(self.ok_list)
            else:
                index = self.ok_list[index]

        for i in range(self.time_steps):
            self.video[i], l = self.dataset[index+i]

        return self.video, []

