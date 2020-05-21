from torch import nn
from torch.autograd import Variable
import torch#.utils.data import DataLoader

def get_gru_initial_state(num_samples,opt):
    return Variable(torch.FloatTensor(num_samples, opt.ngru).normal_())  # m



class Seq2seqGRU(nn.Module):
    def __init__(self,z_dim,opt):
        self.opt = opt
        super(Seq2seqGRU, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(opt.n_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.fc1 =  nn.Sequential(
            nn.Linear(512*4*4, z_dim),
            #nn.ReLU(True),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(opt.ngru,512*4*4),
            #nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128,64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, opt.n_channels, kernel_size=3, stride=1, padding=1,  bias=False),
            #nn.BatchNorm2d(32),
            nn.Tanh(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(opt.ngru,z_dim),
            nn.ReLU(True),
        )

        self.gru = nn.GRU(z_dim,opt.ngru,batch_first=True)
        #self.grucelldecoder = nn.GRU(z_dim,ngru)

    def forward(self, x):
        bs = x.size(0)
        feature = self.encoder(x.reshape(bs*self.opt.T,self.opt.n_channels,64,64))
        z = self.fc1(feature.reshape(bs*self.opt.T,-1))
        z_ = z.reshape(bs,self.opt.T,self.opt.z_dim)
        h = get_gru_initial_state(bs,self.opt).unsqueeze(0).cuda()
        #eos = Variable(torch.ones(bs,z_dim)).cuda()#zero is ok?
        #xhat = torch.empty(T,bs,3*64*64)
        #for t in range(T):
        o,_ = self.gru(z_,h)
        o = self.fc3(o)
        # for t in range(T):
        #     if t==0:
        #         h_hat = self.grucelldecoder(eos,h)#eos?
        #         z_hat = self.fc(h_hat)
        #     else:
        #         h_hat = self.grucelldecoder(z_hat,h_hat)
        #         z_hat = self.fc(h_hat)
        xhat = (self.decoder(o.reshape(bs*self.opt.T,512,4,4)).reshape(bs*self.opt.T,self.opt.n_channels*64*64))
        return xhat,z

