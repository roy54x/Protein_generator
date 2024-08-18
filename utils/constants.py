MAIN_DIR = "D:\python project\data\proteins"

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYUX'
AMINO_ACID_TO_INDEX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 1-indexed for padding

MIN_SIZE = 10
MAX_SIZE = 250
