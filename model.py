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
            nn.Conv2d(in_channels= 16, out_channels= 16, kernel_size=3),
            nn.ReLU()
        )

        #r_in: 5, n_in: 24, j_in: 1, s:1, r_out:7, n_out:22, j_out: 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels=16, kernel_size= 3),
            nn.ReLU()
        )

        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2,2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels= 32, kernel_size= 3),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels= 16, kernel_size= 3),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 16, kernel_size= 3),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 32, kernel_size= 3),
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

dropout_value = 0.1
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) 

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=30, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Dropout(dropout_value)
        ) 

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=10, kernel_size=(3, 3)),
            nn.ReLU(),            
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) 
        self.pool1 = nn.MaxPool2d(2, 2) 

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(1, 1)),
            nn.ReLU(),            
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        ) 

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3)),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) 

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3)),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) 

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) 

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1)),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = self.conv7(x)        

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

    

class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 8, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)         
        )

        self.conv2 =nn.Sequential(
            nn.Conv2d(in_channels= 8, out_channels= 16, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.pool1 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels= 16, kernel_size= (1 ,1)),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels= 16 , out_channels= 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels= 8, out_channels= 8, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )


        self.conv7 =nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels= 16, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size= 5),
            nn.Conv2d(in_channels= 16, out_channels= 10, kernel_size= (1,1)),
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
        x = self.gap1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    

class Model_4(nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 8, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)         
        )

        self.conv2 =nn.Sequential(
            nn.Conv2d(in_channels= 8, out_channels= 16, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.pool1 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels= 16, kernel_size= (1 ,1)),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels= 16 , out_channels= 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels= 8, out_channels= 8, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )


        self.conv7 =nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels= 16, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size= 5),
            nn.Conv2d(in_channels= 16, out_channels= 10, kernel_size= (1,1)),
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
        x = self.gap1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Model_6(nn.Module):
    def __init__(self):
        super(Model_6, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 8, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)         
        )

        self.conv2 =nn.Sequential(
            nn.Conv2d(in_channels= 8, out_channels= 16, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.pool1 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels= 16, kernel_size= (1 ,1)),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels= 16 , out_channels= 8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels= 8, out_channels= 8, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )


        self.conv7 =nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels= 16, kernel_size= 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size= 5),
        ) 

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels= 16, kernel_size= (1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels= 10, kernel_size= (1,1)),
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
        x = self.gap1(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Model_5(nn.Module):
    def __init__(self):
        super(Model_5, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(4)
            
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(8)
        )

        

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(16)
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(16)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(8)
        )


        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(8)
            
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(16)
        )
        

        self.gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5),
            
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1)
        ))

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
      

        x = self.gap1(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)