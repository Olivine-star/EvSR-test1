import torch.nn as nn
import torch
from nett import *
from configs import Config
from torch.utils.data import DataLoader

class NTSR:
    def __init__(self, indata, outdata, indata_sr, conf=Config()):
        self.conf = conf
        self.cuda = conf.cuda
        self.dev = conf.dev if torch.cuda.is_available() else 'cpu'
        self.model = NetT(indata.shape[-1], outdata.shape[-1]).to(self.dev)
        self.learning_rate = self.conf.learning_rate_t
        self.epochs = self.conf.epochs_t
        self.adjust_period = self.conf.adjust_period
        self.adjust_ratio = self.conf.adjust_ratio
        self.indata, self.outdata = indata, outdata
        self.optimizer = None
        self.indata_sr = indata_sr
        self.batch_size = self.conf.batch_size_t if self.conf.batch_size_t != -1 else indata.shape[0]
        self.show_every = self.conf.show_every

    def run(self):
        criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        data_loader = DataLoader(EvData(self.indata, self.outdata), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs + 1):
            self.learning_rate_policy(epoch)
            total_loss = 0
            for indata, outdata in data_loader:
                if self.cuda:
                    indata, outdata = indata.to(torch.float32).to(self.dev), outdata.to(torch.float32).to(self.dev)

                self.optimizer.zero_grad()
                loss = criterion(self.model(indata), outdata)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % self.show_every == 0:
                print('Epoch = ', epoch, ', Train Loss = ', total_loss)

        print('** Finished Training **\n')

    def inference(self):
        self.model.eval()
        outdata_sr = self.model(torch.from_numpy(self.indata_sr).to(torch.float32).to(self.dev))
        return torch.clamp(outdata_sr, min=0, max=1).cpu().detach().numpy()

    def learning_rate_policy(self, epoch):
        if epoch % self.adjust_period == 0 and epoch != 0:
            self.learning_rate /= self.adjust_ratio
            for g in self.optimizer.param_groups:
                g['lr'] = self.learning_rate
            print('Learning Rate Updated = ', self.learning_rate)