import chess
import chess.pgn

from fenstruct import FenStruct

from tqdm import tqdm

import csv

unique_boards = set()

fields = list(FenStruct.__dataclass_fields__.keys())

csvfile = open('data/master.csv', 'w')
csvwriter = csv.DictWriter(csvfile, fieldnames=fields)
csvwriter.writeheader()

def parse_board(board: chess.Board):
    fen = FenStruct.from_board(board)
    fen_str = fen.piece_str + fen.turn_str + fen.castling_str + fen.ep_str

    if fen_str not in unique_boards:
        unique_boards.add(fen_str)
        csvwriter.writerow(fen.__dict__)

def extract_from_pgn(pgn_path: str, game_limit: int):
    pgn = open(pgn_path, encoding="utf-8")

    game_counter = 0

    with tqdm(total=game_limit, desc="Parsing games: ") as pbar:
        while game_counter < game_limit:
            game = chess.pgn.read_game(pgn)
            
            if game is None: break

            board = game.board()

            for move in game.mainline_moves():
                board.push(move)
                parse_board(board)

            game_counter += 1
            pbar.update()
    
    csvfile.close()

extract_from_pgn('data/master_game.pgn', 1e5)
