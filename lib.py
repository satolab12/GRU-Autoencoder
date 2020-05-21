import numpy as np
from sklearn.manifold import TSNE
import os
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import argparse


class ParseGRU():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='../../DATASET/prima/train', help='log directory')
        parser.add_argument('--log_folder', default='./logs/actions/gru-ae', help='log directory')
        parser.add_argument('--batch_size', type=int,default=16)#../DATASET/UCSD/train/
        parser.add_argument('--video_batch', type=int,default=16)
        parser.add_argument('--image_size', default=64)
        parser.add_argument('--T', type=int, default=16, help='checkpoint epoch')
        parser.add_argument('--check_point', type=int, default=10, help='apply SpectralNorm')#SNシますか?
        parser.add_argument('--n_channels', type=int, default=1, help='apply Self-atten')  # Attnシますか?
        parser.add_argument('--num_epochs', type=int, default=100, help='apply Self-atten')
        parser.add_argument('--z_dim', type=int, default=64, help='weight decay')
        parser.add_argument('--ngru', type=int, default=100, help='dimension of latent variable')#512,128,32
        parser.add_argument('--learning_rate', type=int, default=1e-4, help='coefficient of L_prior')#1e-2
        #parser.add_argument('--learning_rate_d', type=int, default=6e-4, help='coefficient of L_prior')  # 1e-2
        parser.add_argument('--cuda', type=bool, default=True, help='weight decay')

        self.args = parser.parse_args()

class Visualizer():
    def __init__(self,opt):
        self.opt = opt

    def plot_loss(self):
        fig, ax = plt.subplots()
        pylab.xlim(0, self.opt.num_epochs )  # *self.len
        pylab.ylim(0, max(self.losses))
        x = np.linspace(0, self.opt.num_epochs)
        ax.plot(x, self.losses, label='loss')#'+','.join(self.opt.dis_loss))#if wanna print type of loss
        ax.legend()
        plt.savefig(os.path.join(self.opt.log_folder, 'loss.pdf'))
        plt.close()


