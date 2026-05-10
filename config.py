import os

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)

RESULTS_DIR = os.path.join(
    OUTPUT_DIR,
    "results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

BATCH_SIZE = 64

EPOCHS = 5

LR = 1e-3

NUM_CLASSES = 10

CONFIDENCE_THRESHOLD = 0.8
