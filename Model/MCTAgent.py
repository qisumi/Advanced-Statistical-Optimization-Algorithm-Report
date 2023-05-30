from collections import deque
import random
from Model.MCTree import TreeNode
import torch
from Model.PolicyValueNet import PolicyValueNet
from utils.FIX_GOMOKU import  GomokuState, Board
from utils.FIX_GOMOKU import gomoku_util
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
class MCTree(object):
    def __init__(self,board_size=15,c_puct=5.0,nsearch=1000) -> None:
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
        actions = state.board.get_legal_action()
        probs = probs[actions]
        if state.board.is_terminal():
            _,win_color = gomoku_util.check_five_in_row(state.board.board_state, state.board.n_inrow)
            if win_color == 'empty':
                val = 0.0
            elif win_color == state.color:
                val = 1.0
            else:
                val = -1.0
        else:
            node.expand(actions,probs)
            
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
        actions = np.array(actions)
        probs = np.log(np.array(nvisits)+1e-6)/temperature
        probs = softmax(probs)
        return actions, probs

    def reset(self):
        self.root = TreeNode(None,1.0)
        
    def step(self,action):
        self.root = self.root.children[action]
        self.root.parent = None
        
class MCTRunner(object):
    def __init__(self, pvnet:PolicyValueNet, board_size = 15,
        eps = 0.25, alpha = 0.03, 
        c_puct=5.0, nsearch=1000, selfplay=False) -> None:
        
        self.pvnet = pvnet
        self.mcttree = MCTree(board_size, c_puct, nsearch)
        self.board_size = board_size
        self.selfplay = selfplay
        self.eps = eps
        self.alpha = alpha
    
    def reset(self):
        self.mcttree.reset()

    def play(self, state, temperature=1e-3, return_data=False):

        probs = np.zeros(self.board_size*self.board_size)
        feat = self.mcttree.get_feature(state.board, state.color)

        a, p = self.mcttree.alpha(state, self.pvnet, temperature)
        probs[a] = p

        action = -1
        if self.selfplay:
            p = (1 - self.eps)*p + \
                self.eps*np.random.dirichlet([self.alpha]*len(a))
            action = np.random.choice(a, p=p)
            self.mcttree.step(action)
        else:
            action = np.random.choice(a, p=p)
            self.mcttree.reset() 

        if return_data:
            return action, feat, probs
        else:
            return action
        
class MCTTrainer(object):
    def __init__(self,device, use_checkpoint:False,board_size=6,n_inrow=4):
        self.board_size = board_size
        self.n_inrow = n_inrow
        self.buffer_size = 10000
        self.c_puct = 5.0
        self.nsearch = 400
        self.temperature = 1e-3
        self.device = device
        self.lr = 1e-3
        self.l2_reg = 1e-4
        self.niter = 5
        self.batch_size = 128
        self.ntrain = 1000

        self.buffer = deque(maxlen=self.buffer_size)
        self.pvnet = PolicyValueNet(self.board_size,device).to(device=device)
        self.idx = 1
        if use_checkpoint:
            self.pvnet.load_state_dict(torch.load(f"checkpoints/"+
                       f"{self.board_size}x{self.board_size}_{self.n_inrow}_pvnet.pth"))
            self.idx = torch.load(f"checkpoints/{self.board_size}x{self.board_size}_{self.n_inrow}_idx.pth") + 1
        # self.pvnet = torch.compile(self.pvnet,mode="max-autotune")
        self.optimizer = torch.optim.Adam(self.pvnet.parameters(), 
            lr=self.lr, weight_decay=self.l2_reg)
        self.mcts_runner = MCTRunner(self.pvnet, self.board_size, 
                c_puct=self.c_puct, nsearch=self.nsearch, selfplay=True)
    def reset_state(self):
        self.state = GomokuState(Board(self.board_size,self.n_inrow), gomoku_util.BLACK)

    def collect_data(self):
        self.reset_state()
        self.mcts_runner.reset()

        feats = []
        probs = []
        players = []
        values = []
        cnt = 0
        while True:
            # print(f"step {cnt+1}"); cnt += 1
            # print(self.state)
            action, feat, prob = self.mcts_runner.play(self.state, self.temperature, True)
            feats.append(feat)
            probs.append(prob)
            players.append(self.state.color)
            self.state = self.state.act(action)

            if self.state.board.is_terminal():
                _, win_color = \
                    gomoku_util.check_five_in_row(self.state.board.board_state,self.state.board.n_inrow)
                if win_color == 'empty':
                    values = [0.0]*len(players)
                else:
                    values = [1.0 if player == win_color else -1.0 for player in players]

                return zip(feats, probs, values)
            
    def data_augment(self, data):
        ret = []
        for feat, prob, value in data:
            for i in range(0, 4):
                feat = np.rot90(feat, i, (1, 2))
                ret.append((feat, prob, value))
                ret.append((feat[:,::-1,:], prob, value))
                ret.append((feat[:,:,::-1], prob, value))
        return ret

    def train_step(self):
        data = self.collect_data()
        data = self.data_augment(data)
        self.buffer.extend(data)

        for idx in range(self.niter):
            feats, probs, values = zip(*random.sample(self.buffer, self.batch_size))
            feats = torch.tensor(np.stack(feats, axis=0)).to(device=self.device)
            probs = torch.tensor(np.stack(probs, axis=0)).to(device=self.device)
            values = torch.tensor(np.stack(values, axis=0)).to(device=self.device)

            p, v = self.pvnet(feats)
            loss = (v - values).pow(2).mean() - (probs*(p + 1e-6).log()).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def train(self):
        print(f"{self.board_size}x{self.board_size}_{self.n_inrow}")
        for idx in tqdm(range(self.idx,self.idx+self.ntrain)):

            self.train_step()
            torch.save(self.pvnet.state_dict(),f"checkpoints/"+
                       f"{self.board_size}x{self.board_size}_{self.n_inrow}_pvnet.pth")
            torch.save(idx,f"checkpoints/{self.board_size}x{self.board_size}_{self.n_inrow}_idx.pth")
            if idx%100==0:
                torch.save(self.pvnet.state_dict(),f"checkpoints/"+
                       f"{self.board_size}x{self.board_size}_{self.n_inrow}_pvnet_{idx}.pth")
