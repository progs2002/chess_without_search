import abc

import argparse

import numpy as np

import chess
import chess.pgn
import chess.engine

import jsonlines

from src.fenstruct import FenStruct
from src.utils import cp_to_win_percent

from tqdm import tqdm

import csv
import pandas as pd

class Extractor:
    def __init__(
        self,
        source_path: str, 
        csv_out_path: str 
    ):
        self.source_path = source_path
        self.csv_out_path = csv_out_path

        self.csv_file = open(self.csv_out_path,'w')
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=list(FenStruct.__dataclass_fields__.keys()) + ['score']
        )

        self.csv_writer.writeheader()

    @abc.abstractmethod 
    def extract(self, game_limit: int):
        raise NotImplementedError

    def _shuffle_csv(self):
        print(f'Shuffling rows of {self.csv_out_path}')
        df = pd.read_csv(self.csv_out_path, dtype=str)
        df = df.sample(n=len(df))
        df.to_csv(self.csv_out_path, index=False)
    

class JsonExtractor(Extractor):
    def __init__(
        self,
        source_path: str, 
        csv_out_path: str, 
        shuffle: bool = True 
    ):
        super().__init__(
            source_path,
            csv_out_path,
        )

        self.shuffle = shuffle

    @classmethod
    def _get_score(cls, cp, mate=False):
        if mate:
            cp = np.sign(cp)*100000 - cp
        return cp_to_win_percent(cp)

    def _parse_board(self, fen, pv):
        is_mate = 'mate' in pv.keys()
        cp = next(iter(pv.values()))

        score = JsonExtractor._get_score(cp, is_mate)
        fen_dict = fen.__dict__
        fen_dict.update({"score": score})

        self.csv_writer.writerow(fen_dict)

    def extract(self, game_limit: int):
        with jsonlines.open(self.source_path, 'r') as f:
            with tqdm(total=game_limit) as pbar:
                for idx, line in enumerate(f.iter()):

                    if idx > game_limit: break

                    board = chess.Board(line["fen"])

                    out = line['evals'][0]['pvs'][0]

                    self._parse_board(FenStruct.from_board(board), out)

                    pbar.update()

        self.csv_file.close()

        if self.shuffle:
            self._shuffle_csv()

class PgnExtractor(Extractor):
    def __init__(
        self,
        source_path: str, 
        csv_out_path: str, 
        shuffle: bool = True 
    ):
        super().__init__(
            source_path,
            csv_out_path
        )

        self.shuffle = shuffle

        self.unique_boards = set()
        self.engine = chess.engine.SimpleEngine.popen_uci(
            '/usr/bin/stockfish'
        )

    def _get_score(self, board: chess.Board) -> chess.engine.Score:
        info = self.engine.analyse(
            board, 
            chess.engine.Limit(time=0.001),
            info=chess.engine.INFO_SCORE
        )

        score= info['score'].white()
        score = score.score(mate_score=100000)
        
        return cp_to_win_percent(score)

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
        pgn = open(self.source_path, encoding="utf-8")

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

        if self.shuffle:
            self._shuffle_csv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", help="add the path to the source pgn or json")
    parser.add_argument('-o', help="add the path to the output csv")
    parser.add_argument("-n", help="number of games", type=int, required=True)
    args = parser.parse_args()

    extractor = Extractor(args.source_path, args.o)
    extractor.extract(args.n)