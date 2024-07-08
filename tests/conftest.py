import pytest
import chess
import random

@pytest.fixture(scope="module")
def board() -> chess.Board:
    board = chess.Board()
    num_moves = random.randint(4,40)
    
    move_counter = 0

    while(True):
        legal_moves = list(board.generate_legal_moves())
        if len(legal_moves) == 0 or move_counter > num_moves:
            break
        
        move = random.choice(legal_moves)
        board.push(move)

        move_counter += 1

    return board