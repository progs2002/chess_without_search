import argparse

import chess
import chess.pgn
import chess.engine

from fenstruct import FenStruct

from tqdm import tqdm

import csv

class Extractor:
    def __init__(self, pgn_path, csv_path):
        self.pgn_path = pgn_path
        self.csv_path = csv_path
        self.csv_file = open(self.csv_path,'w')
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=list(FenStruct.__dataclass_fields__.keys()) + ['score']
        )

        self.csv_writer.writeheader()

        self.engine = chess.engine.SimpleEngine.popen_uci(
            '/usr/bin/stockfish'
        )

        self.unique_boards = set()

    def _get_score(self, board: chess.Board) -> chess.engine.Score:
        info = self.engine.analyse(
            board, 
            chess.engine.Limit(time=0.001),
            info=chess.engine.INFO_SCORE
        )

        score:chess.engine.Score = info['score'].white()

        return score.score(mate_score=100000)


    def _parse_board(self, board: chess.Board):
        fen = FenStruct.from_board(board)

        score = self._get_score(board)

        fen_str = fen.piece_str + fen.castling_str + fen.ep_str

        if fen_str not in self.unique_boards:
            self.unique_boards.add(fen_str)
            fen_dict = fen.__dict__
            fen_dict.update({"score": score})
            self.csv_writer.writerow(fen_dict)

    def extract(self, game_limit: int):
        pgn = open(self.pgn_path, encoding="utf-8")

        game_counter = 0

        with tqdm(total=game_limit, desc="Parsing games: ") as pbar:
            while game_counter < game_limit:
                game = chess.pgn.read_game(pgn)
                
                if game is None: break

                board = game.board()

                for move in game.mainline_moves():
                    board.push(move)
                    self._parse_board(board)

                game_counter += 1
                pbar.update()
        
        self.csv_file.close()
        self.engine.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn_path", help="add the path to the source pgn")
    parser.add_argument('-o', help="add the path to the output csv")
    parser.add_argument("-n", help="number of games", type=int)
    args = parser.parse_args()

    extractor = Extractor(args.pgn_path, args.o)
    extractor.extract(args.n)