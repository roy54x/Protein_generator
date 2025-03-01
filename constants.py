MAIN_DIR = r"/content/drive/My Drive/Colab Notebooks/proteins"
PRETRAINED_MODEL_PATH = r"models\CoordsToSequence\20250216\best_model.pth"

NUM_SAMPLES_IN_DATAFRAME = 100

AMINO_ACIDS = ['<null_0>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E',
               'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
               'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>', '<cath>', '<af2>']
AMINO_ACID_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

MIN_SIZE = 3
MAX_TRAINING_SIZE = 250
MAX_SIZE = 250
BATCH_SIZE = 8

RANDOM_MASK_RATIO = 0.1
SPAN_MASK_RATIO = 0.1
MAX_SPAN_LENGTH = 30

DECAY_RATE = 0.25
