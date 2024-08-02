import numpy as np

import chess
from src.engine import get_moves

b = chess.Board()

turn = 0

while(True):
    if turn%2 == 0:
        #white's turn
        print('white to play')
        mv = input('enter your move ')
        try:
            b.push_san(mv)
        except ValueError:
            mv = input('enter your move again ')
            b.push_san(mv)
    else:
        print('black to play')
        moves, scores = get_moves(b)

        scores = 1 - scores
        idx = np.argmax(scores)
        mv = moves[idx]

        b.push(mv)
        print(mv)
    print(b)
    turn += 1
