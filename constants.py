MAIN_DIR = r"C:\Users\RoyIlani\Desktop\personal\proteins"
PRETRAINED_MODEL_PATH = r"models\CoordsToLatentSpace\20241231\best_model.pth"

NUM_SAMPLES_IN_DATAFRAME = 100

AMINO_ACIDS = ['<null_0>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E',
               'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
               'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>', '<cath>', '<af2>']
AMINO_ACID_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

MIN_SIZE = 3
MAX_TRAINING_SIZE = 250
MAX_SIZE = 250
BATCH_SIZE = 16

DECAY_RATE = 0.25
