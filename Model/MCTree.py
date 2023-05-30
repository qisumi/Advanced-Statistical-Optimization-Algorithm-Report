import numpy as np

class TreeNode(object):
    def __init__(self,parent,prior) -> None:
        self.parent:TreeNode = parent # 父节点
        self.prior:float = prior # 先验概率
        
        self.Q:float = 0 # 价值函数
        self.N:int = 0 # 访问次数
        
        self.children:dict[float,TreeNode] = {}
        
    def score(self,c_puct):
        # PUCT分数
        sqrt_sum = np.sqrt(
            np.sum([
                node.N for node in self.parent.children.values()
            ])
        )
        return self.Q+c_puct*self.prior*sqrt_sum/(1+self.N)
    
    def update(self,qval):
        self.Q = self.Q*self.N+qval
        self.N+=1
        self.Q = self.Q/self.N
    
    def backup(self,qval):
        # 回溯
        self.update(qval)
        if self.parent:
            self.parent.backup(-qval)
            
    def select(self,c_puct):
        # PUCT选择
        return self.children[np.argmax([
            node.score(c_puct) for node in self.children.values()
        ])]
        
    def expand(self,actions,priors):
        # 扩展
        for action,prior in zip(actions,priors):
            if action not in self.children:
                self.children[action] = TreeNode(self,prior)
                
    @property
    def is_root(self):
        return self.parent is None
    
    @property
    def is_leaf(self):
        return len(self.children) == 0