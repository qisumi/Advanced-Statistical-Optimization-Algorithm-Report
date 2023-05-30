import numpy as np
import gym
from gym import spaces
from gym import error
from gym.utils import seeding
from six import StringIO
import sys, os
import six

from gym_gomoku.envs.util import make_random_policy
from gym_gomoku.envs.util import make_beginner_policy
from gym_gomoku.envs.util import make_medium_policy
from gym_gomoku.envs.util import make_expert_policy

class GomokuUtil(object):
    
    def __init__(self):
        # default setting
        self.BLACK = 'black'
        self.WHITE = 'white'
        self.color = [self.BLACK, self.WHITE]
        self.color_dict = {'empty': 0, 'black': 1, 'white': 2}
        self.color_dict_rev = {v: k for k, v in self.color_dict.items()}
        self.color_shape = {0: '.', 1: 'X', 2: 'O'}
    
    def other_color(self, color):
        '''Return the opositive color of the current player's color
        '''
        assert color in self.color, 'Invalid player color'
        opposite_color = self.color[0] if color == self.color[1] else self.color[1]
        return opposite_color
    
    def iterator(self, board_state):
        ''' Iterator for 2D list board_state
            Return: Row, Column, diagnoal, list of coordinate tuples, [(x1, y1), (x2, y2), ...,()], (6n-2-16) lines
        '''
        list = []
        size = len(board_state)
        
        # row
        for i in range(size): # [(i,0), (i,1), ..., (i,n-1)]
            list.append([(i, j) for j in range(size)])
        
        # column
        for j in range(size):
            list.append([(i, j) for i in range(size)])
        
        # diagonal: left triangle
        for k in range(size):
            # lower_line consist items [k][0], [k-1][1],...,[0][k]
            # upper_line consist items [size-1][size-1-k], [size-1-1][size-1-k +1],...,[size-1-k][size-1]
            lower_line = [((k-k1), k1) for k1 in range(k+1)]
            upper_line = [((size-1-k2), (size-k-1+k2)) for k2 in range(k+1)]
            if (k == (size-1)): # one diagnoal, lower_line same as upper_line
                list.append(lower_line)
            else :
                if (len(lower_line)>=5):
                    list.append(lower_line)
                if (len(upper_line)>=5):
                    list.append(upper_line)
        
        # diagonal: right triangle
        for k in range(size):
            # lower_line consist items [0][k], [1][k+1],...,[size-1-k][size-1]
            # upper_line consist items [k][0], [k+1][1],...,[size-1][size-1-k]
            lower_line = [(k1, k + k1) for k1 in range(size-k)]
            upper_line = [(k + k2, k2) for k2 in range(size-k)]
            if (k == 0): # one diagnoal, lower_line same as upper_line
                list.append(lower_line)
            else :
                if (len(lower_line)>=5):
                    list.append(lower_line)
                if (len(upper_line)>=5):
                    list.append(upper_line)
        
        for line in list:
            yield line
    
    def value(self, board_state, coord_list):
        ''' Fetch Value from 2D list with coord_list
        '''
        val = []
        for (i,j) in coord_list:
            val.append(board_state[i][j])
        return val
    
    def check_five_in_row(self, board_state, n_inrow):
        ''' Args: board_state 2D list
            Return: exist, color
        '''
        size = len(board_state)
        black_pattern = [self.color_dict[self.BLACK] for _ in range(n_inrow)] # [1,1,1,1,1]
        white_pattern = [self.color_dict[self.WHITE] for _ in range(n_inrow)] # [2,2,2,2,2]
        
        exist_final = False
        color_final = "empty"
        black_win, _ = self.check_pattern(board_state, black_pattern)
        white_win, _ = self.check_pattern(board_state, white_pattern)
        
        if (black_win and white_win):
            raise error.Error(f'Both Black and White has {n_inrow}-in-row, rules conflicts')
        # Check if there is any one party wins
        if not (black_win or white_win):
            return exist_final, "empty"
        else:
            exist_final = True
        if (black_win):
            return exist_final, self.BLACK
        if (white_win):
            return exist_final, self.WHITE
    
    def check_board_full(self, board_state):
        is_full = True
        size = len(board_state)
        for i in range(size):
            for j in range(size):
                if (board_state[i][j]==0):
                    is_full = False
                    break
        return is_full
    
    def check_pattern(self, board_state, pattern):
        ''' Check if pattern exist in the board_state lines,
            Return: exist: boolean
                    line: coordinates that contains the patterns
        '''
        exist = False
        pattern_found = [] # there maybe multiple patterns found
        for coord in self.iterator(board_state):
            line_value = self.value(board_state, coord)
            if (self.is_sublist(line_value, pattern)):
                exist = True
                pattern_found.append(coord)
        return exist, pattern_found
    
    def check_pattern_index(self, board_state, pattern):
        '''Return the line contains the pattern, and its start position index of the pattern
        '''
        start = -1
        startlist = []
        exist_patttern, lines = self.check_pattern(board_state, pattern)
        if (exist_patttern):
            for line in lines:
                start = self.index(self.value(board_state, line), pattern)
                startlist.append(start)
            return lines, startlist  # line: list[list[(x1, y1),...]], startlist: list[int]
        else: # pattern not found
            return None, startlist
    
    def is_sublist(self, list, sublist):
        l1 = len(list)
        l2 = len(sublist)
        is_sub = False
        for i in range(l1):
            curSub = list[i: min(i+l2, l1)]
            if (curSub == sublist): # check list equal
                is_sub = True
                break
        return is_sub
    
    def index(self, list, sublist):
        ''' Return the starting index of the sublist in the list
        '''
        idx = - 1
        l1 = len(list)
        l2 = len(sublist)
        
        for i in range(l1):
            curSub = list[i: min(i+l2, l1)]
            if (curSub == sublist): # check list equal
                idx = i
                break
        return idx

gomoku_util = GomokuUtil()

# Rules from Wikipedia: Gomoku is an abstract strategy board game, Gobang or Five in a Row, it is traditionally played with Go pieces (black and white stones) on a go board with 19x19 or (15x15) 
# The winner is the first player to get an unbroken row of five stones horizontally, vertically, or diagonally. (so-calle five-in-a row)
# Black plays first if white did not win in the previous game, and players alternate in placing a stone of their color on an empty intersection.

class GomokuState(object):
    '''
    Similar to Go game, Gomoku state consists of a current player and a board.
    Actions are exposed as integers in [0, num_actions), which is to place stone on empty intersection
    '''
    def __init__(self, board, color):
        '''
        Args:
            board: current board
            color: color of current player
        '''
        assert color in ['black', 'white'], 'Invalid player color'
        self.board, self.color = board, color
    
    def act(self, action):
        '''
        Executes an action for the current player
        
        Returns:
            a new GomokuState with the new board and the player switched
        '''
        return GomokuState(self.board.play(action, self.color), gomoku_util.other_color(self.color))
    
    def __repr__(self):
        '''stream of board shape output'''
        # To Do: Output shape * * * o o
        return 'To play: {}\n{}'.format(six.u(self.color), self.board.__repr__())

# Sampling without replacement Wrapper 
# sample() method will only sample from valid spaces
class DiscreteWrapper(spaces.Discrete):
    def __init__(self, n):
        self.n = n
        self.valid_spaces = list(range(n))
    
    def sample(self):
        '''Only sample from the remaining valid spaces
        '''
        if len(self.valid_spaces) == 0:
            print ("Space is empty")
            return None
        np_random, _ = seeding.np_random()
        randint = np_random.randint(len(self.valid_spaces))
        return self.valid_spaces[randint]
    
    def remove(self, s):
        '''Remove space s from the valid spaces
        '''
        if s is None:
            return
        if s in self.valid_spaces:
            self.valid_spaces.remove(s)
        else:
            print ("space %d is not in valid spaces" % s)

class Board(object):
    '''
    Basic Implementation of a Go Board, natural action are int [0,board_size**2)
    '''
    
    def __init__(self, board_size, n_inrow):
        self.size = board_size
        self.board_state = [[gomoku_util.color_dict['empty']] * board_size for i in range(board_size)] # initialize board states to empty
        self.move = 0                 # how many move has been made
        self.last_coord = (-1,-1)     # last action coord
        self.last_action = None       # last action made
        self.n_inrow = n_inrow
    
    def coord_to_action(self, i, j):
        ''' convert coordinate i, j to action a in [0, board_size**2)
        '''
        a = i * self.size + j # action index
        return a
    
    def action_to_coord(self, a):
        coord = (a // self.size, a % self.size)
        return coord
    
    def get_legal_move(self):
        ''' Get all the next legal move, namely empty space that you can place your 'color' stone
            Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
        '''
        legal_move = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    legal_move.append((i, j))
        return legal_move
    
    def get_legal_action(self):
        ''' Get all the next legal action, namely empty space that you can place your 'color' stone
            Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
        '''
        legal_action = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    legal_action.append(self.coord_to_action(i, j))
        return legal_action
    
    def copy(self, board_state):
        '''update board_state of current board values from input 2D list
        '''
        input_size_x = len(board_state)
        input_size_y = len(board_state[0])
        assert input_size_x == input_size_y, 'input board_state two axises size mismatch'
        assert len(self.board_state) == input_size_x, 'input board_state size mismatch'
        for i in range(self.size):
            for j in range(self.size):
                self.board_state[i][j] = board_state[i][j]
    
    def play(self, action, color):
        '''
            Args: input action, current player color
            Return: new copy of board object
        '''
        b = Board(self.size, self.n_inrow)
        b.copy(self.board_state) # create a board copy of current board_state
        b.move = self.move
        
        coord = self.action_to_coord(action)
        # check if it's legal move
        if (b.board_state[coord[0]][coord[1]] != 0): # the action coordinate is not empty
            raise error.Error("Action is illegal, position [%d, %d] on board is not empty" % ((coord[0]+1),(coord[1]+1)))
        
        b.board_state[coord[0]][coord[1]] = gomoku_util.color_dict[color]
        b.move += 1 # move counter add 1
        b.last_coord = coord # save last coordinate
        b.last_action = action
        return b
    
    def is_terminal(self):
        exist, color = gomoku_util.check_five_in_row(self.board_state, self.n_inrow)
        is_full = gomoku_util.check_board_full(self.board_state)
        if (is_full): # if the board if full of stones and no extra empty spaces, game is finished
            return True
        else:
            return exist
    
    def __repr__(self):
        ''' representation of the board class
            print out board_state
        '''
        out = ""
        size = len(self.board_state)
        
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:size]
        numbers = list(range(1, 100))[:size]
        
        label_move = "Move: " + str(self.move) + "\n"
        label_letters = "     " + " ".join(letters) + "\n"
        label_boundry = "   " + "+-" + "".join(["-"] * (2 * size)) + "+" + "\n"
        
        # construct the board output
        out += (label_move + label_letters + label_boundry)
        
        for i in range(size-1,-1,-1):
            line = ""
            line += (str("%2d" % (i+1)) + " |" + " ")
            for j in range(size):
                # check if it's the last move
                line += gomoku_util.color_shape[self.board_state[i][j]]
                if (i,j) == self.last_coord:
                    line += ")"
                else:
                    line += " "
            line += ("|" + "\n")
            out += line
        out += (label_boundry + label_letters)
        return out
    
    def encode(self):
        '''Return: np array
            np.array(board_size, board_size): state observation of the board
        '''
        img = np.array(self.board_state) # shape [board_size, board_size]
        return img