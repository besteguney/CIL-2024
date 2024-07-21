# Constants
ROOT_PATH = "ethz-cil-road-segmentation-2024"
PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 24  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road
N_ESTIMATORS = 3

# Model Parameters
N_EPOCHS = 1000
BATCH_SIZE = 4
RESIZE = 256
