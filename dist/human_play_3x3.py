from utils.FIX_GOMOKU import GomokuState,Board,gomoku_util
from Model import PolicyValueNet
from Model.MCTAgent import MCTRunner
import numpy as np
import torch

board_size = 3
n_inrow = 3

# 加载AI模型
model = PolicyValueNet(board_size)
print("loading model iter:"+str(torch.load(''+
                 f"{board_size}x{board_size}_{n_inrow}_idx.pth")))
model.load_state_dict(torch.load(f""+
                       f"{board_size}x{board_size}_{n_inrow}_pvnet.pth",map_location=torch.device('cpu')))
AIPlayer = MCTRunner(model, board_size, c_puct=5.0, nsearch=400, selfplay=False)

# 特征提取函数
def get_feature(board:Board, player='black') -> np.ndarray:
    feat = board.encode()
    feat1 = (feat == 1).astype(np.float32)
    feat2 = (feat == 2).astype(np.float32)
    feat3 = np.zeros((board_size, board_size)).astype(np.float32)
    if board.last_action is not None:
        x, y = board.action_to_coord(board.last_action)
        feat3[x, y] = 1.0
    if player == 'white':
        feat4 = np.zeros((board_size, board_size)).astype(np.float32)
        return np.stack([feat1, feat2, feat3, feat4], axis=0)
    elif player == 'black':
        feat4 = np.ones((board_size, board_size)).astype(np.float32)
        return np.stack([feat1, feat2, feat3, feat4], axis=0)

def play_game():
    # 初始化游戏状态和棋盘
    board = Board(board_size,n_inrow)
    state = GomokuState(board,gomoku_util.BLACK)
    current_player = 1
    
    while not state.board.is_terminal():
        if current_player == 1:
            # 使用AI模型作为玩家1的决策
            action = get_ai_move(state)
            state = state.act(action)
            print("AI's move:", action)
            print(state.board)
            current_player = 0
        else:
            # 玩家2从控制台输入决策
            action = get_human_move(state)
            state = state.act(action)
            print("Player's move:", action)
            print(state.board)
            current_player = 1

    # 游戏结束，打印胜者
    _,win_color = gomoku_util.check_five_in_row(state.board.board_state,n_inrow)
    if win_color == gomoku_util.BLACK:
        print("AI wins!")
    elif win_color == gomoku_util.WHITE:
        print("Player wins!")
    else:
        print("It's a draw!")

def get_ai_move(state:GomokuState):
    action = AIPlayer.play(state)
    return action
    
def get_human_move(state:GomokuState):
    board:Board = state.board
    while True:
        try:
            move = str(input("Enter your move like A1,C2 or something: "))
            print(move)
            move = (int(move[1])-1,ord(move[0])-ord('A'))
            print(move)
            if move in board.get_legal_move():
                return board.coord_to_action(move[0],move[1])
            else:
                print("Invalid move. Try again.")
        except ValueError:
            print("Invalid input. Try again.")

# 开始游戏
play_game()