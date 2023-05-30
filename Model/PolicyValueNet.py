import torch
from torch import nn


class PolicyValueNet(nn.Module):
    def __init__(self,board_size) -> None:
        super().__init__()
        
        # 特征提取
        self.feature_net = nn.Sequential(
            nn.Conv2d(4,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        )
        
        # 策略生成
        self.policy_net = nn.Sequential(
            nn.Conv2d(128,4,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*board_size*board_size,board_size*board_size),
        )
        
        # 价值函数
        self.value_net = nn.Sequential(
            nn.Conv2d(128,2,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2*board_size*board_size,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        
    def forward(self,x)->tuple[torch.Tensor,torch.Tensor]:
        feature = self.feature_net(x)
        prob = self.policy_net(feature)
        prob = torch.softmax(prob,dim=-1)
        value = self.value_net(feature)
        value = torch.tanh(value)
        return prob,value
    
    def evaluate(self,x) -> tuple[torch.Tensor,torch.Tensor]:
        with torch.no_grad():
            prob,value = self.forward(x)
        return prob.squeeze(),value.squeeze()