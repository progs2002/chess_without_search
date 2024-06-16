import chess
import numpy as np 
import chess.pgn
from tqdm import tqdm
from stockfish import Stockfish

stockfish_client = Stockfish('/usr/bin/stockfish', parameters={"Threads": 1, "Minimum Thinking Time": 50})

def get_unique_boards(pgn_path, game_limit:int=1e5):
    pgn = open(pgn_path, encoding="utf-8")

    unique_boards = set()
    game_counter = 0

    with tqdm(total=game_limit, desc="Parsing games ") as pbar:
        while game_counter < game_limit:
            game = chess.pgn.read_game(pgn)
            
            if game is None: break

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                unique_boards.add(board.fen())
            game_counter += 1
            pbar.update()

    return unique_boards

def generate_action_pairs(board_fen, idx):

    with open(f'data/unique_boards/board_{idx}', 'w') as f:
        f.write(f'{board_fen}\n')
        board = chess.Board(board_fen)
        for move in tqdm(board.generate_legal_moves(), desc=f"Evaluating legal moves for board {idx}", leave=False):
            board.push(move)
            stockfish_client.set_fen_position(board.fen())
            eval = stockfish_client.get_evaluation()
            f.write(f'{move}\t{eval}\n')
            board.pop()

unique_boards = list(get_unique_boards('data/master_game.pgn', 2))

with tqdm(total=len(unique_boards), desc="Generating action pairs for boards") as pbar:
    for idx, board in enumerate(unique_boards):
        generate_action_pairs(board, idx)
        pbar.update()