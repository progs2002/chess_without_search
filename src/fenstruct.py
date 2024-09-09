'''
convert FEN to fixed length string of 76 characters (pad with .)
ASCII code of each character us one token 
'''

import dataclasses
import chess

from typing import Self

@dataclasses.dataclass(unsafe_hash=True)
class FenStruct:
    
    piece_str: str
    turn_str: str
    castling_str: str
    ep_str: str
    hc_str: str
    fc_str: str

    @classmethod
    def _get_piece_str(cls, board: chess.Board) -> str:
        #code influenced by https://github.com/niklasf/python-chess/blob/59cadb1f06a4a36499e006e2534289e52cb36c2f/chess/__init__.py#L1079
        builder = []
        for square in chess.SQUARES_180:
            piece = board.piece_at(square)

            if not piece:
                builder.append('.')
            else:
                builder.append(piece.symbol())

        return "".join(builder)
    
    @classmethod
    def from_board(cls, board: chess.Board) -> Self:
        piece_str = cls._get_piece_str(board)
        assert len(piece_str) == 64

        turn_str = "w" if board.turn == chess.WHITE else "b"

        castling_str = board.castling_xfen()
        if castling_str == '-':
            castling_str = ''
        castling_str = castling_str.ljust(4, ".")

        ep_square = board.ep_square 
        # ep_str = chess.SQUARE_NAMES[ep_square] if ep_square is not None else "-"
        ep_str = chess.SQUARE_NAMES[ep_square] if ep_square is not None else ".."
        
        hc_str = str(board.halfmove_clock).ljust(3, ".")
        fc_str = str(board.fullmove_number).ljust(3, ".")

        return cls(
            piece_str,
            turn_str,
            castling_str,
            ep_str,
            hc_str,
            fc_str
        )
        

if __name__ == "__main__":
    board = chess.Board()
    board.push(chess.Move.from_uci("g1f3"))
    fen_b1 = FenStruct.from_board(board)
    print(fen_b1)