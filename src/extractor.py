import asyncio

from typing import Optional

import chess
import chess.pgn
import chess.engine

from fenstruct import FenStruct

from tqdm import tqdm

import csv

unique_boards = set()

fields = list(FenStruct.__dataclass_fields__.keys()) + ['score']

csvfile = open('data/master_demo_2.csv', 'w')
csvwriter = csv.DictWriter(csvfile, fieldnames=fields)
csvwriter.writeheader()

engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish") 

def get_score(board: chess.Board) -> chess.engine.Score:
    info = engine.analyse(
        board, 
        chess.engine.Limit(time=0.001),
        info=chess.engine.INFO_SCORE
    )

    score:chess.engine.Score = info['score'].white()

    return score.score(mate_score=100000)


def parse_board(board: chess.Board):
    fen = FenStruct.from_board(board)

    score = get_score(board)

    fen_str = fen.piece_str + fen.castling_str + fen.ep_str

    if fen_str not in unique_boards:
        unique_boards.add(fen_str)
        fen_dict = fen.__dict__
        fen_dict.update({"score": score})
        csvwriter.writerow(fen_dict)

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

extract_from_pgn('data/master_game.pgn', 100000)
engine.quit()