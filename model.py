import torch.nn as nn
import torch.nn.functional as F

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()

        #r_in: 1, n_in: 28, j_in: 1, s:1, r_out:3, n_out:26, j_out: 1 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 16, kernel_size = 3),
            nn.ReLU()
        )

        #r_in: 3, n_in: 26, j_in: 1, s:1, r_out:5, n_out:24, j_out: 1 
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 32, kernel_size=3),
            nn.ReLU()
        )

        #r_in: 5, n_in: 24, j_in: 1, s:1, r_out:7, n_out:22, j_out: 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels= 32, out_channels=64, kernel_size= 3),
            nn.ReLU()
        )

        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2,2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels= 64, kernel_size= 3),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels= 64, kernel_size= 3),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels= 32, kernel_size= 3),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 3),
            nn.ReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels= 32, out_channels= 16, kernel_size= 3),
            
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(-1, 16)
        return F.log_softmax(x, dim=-1)


class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 16, kernel_size= 3),
            nn.ReLU()
        )

        self.conv2 =nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 16, kernel_size= 3),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 16, kernel_size=3),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels= 16, kernel_size= (1,1)),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels= 16 , out_channels= 16, kernel_size=3),
            nn.ReLU()
        )

