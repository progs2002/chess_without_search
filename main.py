import argparse
from src.extractor import Extractor

parser = argparse.ArgumentParser()
parser.add_argument("pgn_path", help="add the path to the source pgn")
parser.add_argument('-o', help="add the path to the output csv")
parser.add_argument("-n", help="number of games", type=int, required=True)
args = parser.parse_args()

extractor = Extractor(args.pgn_path, args.o)
extractor.extract(args.n)