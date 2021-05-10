import torch
from torch import nn


class RepMLP(nn.Module):
    def __init__(self,C,H,W,h,w,fc1_fc2_reduction=1,fc3_groups=8,repconv_kernels=None,deploy=False):
        super().__init__()
        self.C=C
        self.O=O
        self.H=H
        self.W=W
        self.h=h
        self.w=w
        self.fc1_fc2_reduction=fc1_fc2_reduction
        self.repconv_kernels=repconv_kernels
        self.h_part=H//h
        self.w_part=W//w
        self.deploy=deploy

        # make sure H,W can divided by h,w respectively
        assert H%h==0
        assert W%w==0

        self.is_global_perceptron= (H!=h) or (W!=w)
        ### global perceptron
        if(self.is_global_perceptron):
            if(not self.deploy):
                self.avg=nn.Sequential(
                    nn.AvgPool2d(kernel_size=(self.h,self.w)),
                    nn.BatchNorm2d(num_features=C)
                )
            else:
                self.avg=nn.AvgPool2d(kernel_size=(self.h,self.w))
            hidden_dim=self.C//self.fc1_fc2_reduction
            self.fc1_fc2=nn.Sequential(
                nn.Linear(C*self.h_part*self.w_part,hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim,C*self.h_part*self.w_part)
            )

        self.fc3=nn.Conv2d(self.C*self.h*self.w,self.O*self.h*self.w,kernel_size=1,groups=fc3_groups,bias=self.deploy)
        self.fc3_bn=nn.Identity() if self.deploy else nn.BatchNorm2d(self.O*self.h*self.w)
        
        if not self.deploy and self.repconv_kernels is not None:
            for k in self.repconv_kernels:
                repconv=nn.Sequential(
                    nn.Conv2d(self.C,self.O,kernel_size=k,padding=(k-1)//2),
                    nn.BatchNorm2d(self.O)
                )
                self.__setattr__('repconv{}'.format(k),repconv)
                



            
        

    def forward(self,x) :
        ### global partition
        if(self.is_global_perceptron):
            input=x
            v=self.avg(x) #bs,C,h_part,w_part
            v=v.reshape(-1,self.C*self.h_part*self.w_part) #bs,C*h_part*w_part
            v=self.fc1_fc2(v) #bs,C*h_part*w_part
            v=v.reshape(-1,self.C,self.h_part,1,self.w_part,1) #bs,C,h_part,w_part
            input=input.reshape(-1,self.C,self.h_part,self.h,self.w_part,self.w) #bs,C,h_part,h,w_part,w
            input=v+input
        else:
            input=x.view(-1,self.C,self.h_part,self.h,self.w_part,self.w) #bs,C,h_part,h,w_part,w
        partition=input.permute(0,2,4,1,3,5) #bs,h_part,w_part,C,h,w

        ### partition partition
        fc3_out=partition.reshape(-1,self.C*self.h*self.w,1,1) #bs*h_part*w_part,C*h*w,1,1
        fc3_out=self.fc3_bn(self.fc3(fc3_out)) #bs*h_part*w_part,O*h*w,1,1
        fc3_out=fc3_out.reshape(-1,self.h_part,self.w_part,O,self.h,self.w) #bs,h_part,w_part,O,h,w

        ### local perceptron
        if(self.repconv_kernels is not None):
            conv_input=partition.reshape(-1,self.C,self.h,self.w) #bs*h_part*w_part,C,h,w
            conv_out=0
            for k in self.repconv_kernels:
                repconv=self.__getattr__('repconv{}'.format(k))
                conv_out+=repconv(conv_input) ##bs*h_part*w_part,O,h,w
            conv_out=conv_out.view(-1,self.h_part,self.w_part,self.O,self.h,self.w) #bs,h_part,w_part,O,h,w
            fc3_out+=conv_out
        fc3_out=fc3_out.permute(0,3,1,4,2,5)#bs,O,h_part,h,w_part,w
        fc3_out=fc3_out.reshape(-1,self.C,self.H,self.W) #bs,O,H,W


        return fc3_out



if __name__ == '__main__':
    N=4
    C=512
    O=1024
    H=14
    W=14
    h=7
    w=7
    fc1_fc2_reduction=1
    fc3_groups=8
    repconv_kernels=[1,3,5,7]
    repmlp=RepMLP(C,H,W,h,w,fc1_fc2_reduction,fc3_groups,repconv_kernels=repconv_kernels)
    x=torch.randn(N,C,H,W)
    y=repmlp(x)
    print(y.shape)
    

