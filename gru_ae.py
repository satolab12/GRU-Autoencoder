import numpy as np
import torch
from torch import nn
from lib import ParseGRU,Visualizer,Seq_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets, transforms
from torchvision.utils import save_image
from network import Seq2seqGRU

import os


parse = ParseGRU()
opt = parse.args
len_test = 3
autoencoder = Seq2seqGRU(opt.z_dim,opt)
autoencoder.train()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(),
                             lr=opt.learning_rate,
                             weight_decay=1e-5)

if opt.cuda:
    autoencoder.cuda()

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((opt.image_size,opt.image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
    ])

dataset_ = datasets.ImageFolder(opt.dataset, transform=transform)  # has all image shape,[data,label]
data_ = Seq_Dataset(dataset_, opt.T,shuffle=True)  # data_ has 2870 shape
video_loader = DataLoader(data_, batch_size=opt.batch_size, shuffle=True)  # if shuffle True index is ramdom

losses = np.zeros(opt.num_epochs)
visual = Visualizer(opt)

for epoch in range(opt.num_epochs):
    i = 0
    for data,label in video_loader:

        x = data.transpose(2,1).reshape(-1, opt.T,opt.n_channels*opt.image_size*opt.image_size)

        if opt.cuda:
            x = Variable(x).cuda()
        else:
            x = Variable(x)

        xhat,z = autoencoder(x)
        xhat = xhat.reshape(-1, opt.T,opt.n_channels*opt.image_size*opt.image_size)

        # 出力画像（再構成画像）と入力画像の間でlossを計算
        loss = mse_loss(xhat, x)
        losses[epoch] = losses[epoch] * (i / (i + 1.)) + loss.data * (1. / (i + 1.))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss: {:.4f}'.format(
        epoch + 1,
        opt.num_epochs,
        loss))
    visual.losses = losses
    visual.plot_loss()

    if epoch % opt.check_point == 0:
        j = 0
        tests = x[:len_test].reshape(-1,opt.T,opt.n_channels,opt.image_size,opt.image_size)
        recon = xhat[:len_test].reshape(-1,opt.T,opt.n_channels,opt.image_size,opt.image_size)

        for i in range(len_test):
            save_image((tests[i]/2+0.5), os.path.join(opt.log_folder + '/generated_videos', "real_epoch{}_no{}.png" .format(epoch,i)))
            save_image((recon[i]/2+0.5), os.path.join(opt.log_folder+'/generated_videos', "recon_epoch{}_no{}.png" .format(epoch,i)))
            #torch.save(autoencoder.state_dict(), os.path.join('./weights', 'G_epoch{:04d}.pth'.format(epoch+1)))


