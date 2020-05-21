import torch
import sys
import numpy as np

from torch.utils.data import Dataset

class Seq_Dataset(Dataset):
    def __init__(self, datasets ,time_steps,slide=True,shuffle=True):
        self.dataset = datasets
        self.slide = slide
        self.shuffle = shuffle
        print('init!')
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

