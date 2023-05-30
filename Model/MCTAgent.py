from Model.MCTree import TreeNode
import torch
from Model.PolicyValueNet import PolicyValueNet
from gym_gomoku.envs.gomoku import  GomokuState, Board
from gym_gomoku.envs.util import gomoku_util
import numpy as np

class MCTree(object):
    def __init__(self,board_size=15,c_puct=5.0,nsearch=2000) -> None:
        self.board_size:int = board_size
        self.root:TreeNode = TreeNode(None,1.0)
        self.c_puct:float = c_puct
        self.nsearch:int = nsearch
         
    def get_feature(self, board:Board, player) -> np.ndarray:
        feat = board.encode()
        feat1 = (feat == 1).astype(np.float32)
        feat2 = (feat == 2).astype(np.float32)
        feat3 = np.zeros((self.board_size, self.board_size)).astype(np.float32)
        if board.last_action is not None:
            x, y = board.action_to_coord(board.last_action)
            feat3[x, y] = 1.0
        if player == 'white':
            feat4 = np.zeros((self.board_size, self.board_size)).astype(np.float32)
            return np.stack([feat1, feat2, feat3, feat4], axis=0)
        elif player == 'black':
            feat4 = np.ones((self.board_size, self.board_size)).astype(np.float32)
            return np.stack([feat1, feat2, feat3, feat4], axis=0)
        
    def mct_search(self,state:GomokuState,pvnet:PolicyValueNet):
        
        # 定义根节点
        node = self.root
        
        # 搜索
        while not node.is_leaf:
            action, node = node.select(self.c_puct)
            state = state.act(action)
            
        # 扩展、求值
        feature = self.get_feature(state.board, state.color)
        feature = torch.tensor(feature).unsqueeze(0)
        probs,val = pvnet.evaluate(feature)
        actions = state.board.get_legal_actions()
        probs = probs[actions]
        if state.board.is_terminal():
            _,win_color = gomoku_util.check_five_in_row(state.board.board_state)
            if win_color == 'empty':
                val = 0.0
            elif win_color == state.color:
                val = 1.0
            else:
                val = -1.0
        else:
            node.expand(action,probs)
            
        # 备份
        node.backup(-val)
        
    def alpha(self,state:GomokuState,pvnet:PolicyValueNet, temperature = 1e-3):
        # 计算子节点概率
        for _ in range(self.nsearch):
            self.mct_search(state,pvnet)
        node_info = [
            (action,node.N)
            for action, node in self.root.children.items()
        ]
        actions,nvisits = zip(*node_info)
        actions = torch.tensor(actions)
        probs = torch.log(torch.tensor(nvisits)+1e-6/temperature)
        probs = torch.softmax(probs)
        return actions,probs

    def reset(self):
        self.root = TreeNode(None,1.0)
        
    def step(self,action):
        self.root = self.root.children[action]
        self.root.parent = None
        
class MCTRunner(object):
    pass