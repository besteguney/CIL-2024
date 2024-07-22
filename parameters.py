#Constants
ROOT_PATH = "data"
PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road
N_ESTIMATORS = 3

# Model Parameters
N_EPOCHS = 20
BATCH_SIZE = 4
RESIZE = 384



ENCODERS = {
    "resnet18": 11_000_000,
    "resnet34": 21_000_000,
    "resnet50": 23_000_000,
    "resnet101": 42_000_000,
    "resnet152": 58_000_000,
    "resnext50_32x4d": 22_000_000,
    "resnext101_32x8d": 86_000_000,
    "resnext101_32x16d": 191_000_000,
    "resnext101_32x32d": 466_000_000,
    "resnext101_32x48d": 826_000_000,
    "dpn68": 11_000_000,
    "dpn68b": 11_000_000,
    "dpn92": 34_000_000,
    "dpn98": 58_000_000,
    "dpn107": 84_000_000,
    "dpn131": 76_000_000,
    "vgg11": 9_000_000,
    "vgg11_bn": 9_000_000,
    "vgg13": 9_000_000,
    "vgg13_bn": 9_000_000,
    "vgg16": 14_000_000,
    "vgg16_bn": 14_000_000,
    "vgg19": 20_000_000,
    "vgg19_bn": 20_000_000,
    "senet154": 113_000_000,
    "se_resnet50": 26_000_000,
    "se_resnet101": 47_000_000,
    "se_resnet152": 64_000_000,
    "se_resnext50_32x4d": 25_000_000,
    "se_resnext101_32x4d": 46_000_000,
    "densenet121": 6_000_000,
    "densenet169": 12_000_000,
    "densenet201": 18_000_000,
    "densenet161": 26_000_000,
    "inceptionresnetv2": 54_000_000,
    "inceptionv4": 41_000_000,
    "efficientnet-b0": 4_000_000,
    "efficientnet-b1": 6_000_000,
    "efficientnet-b2": 7_000_000,
    "efficientnet-b3": 10_000_000,
    "efficientnet-b4": 17_000_000,
    "efficientnet-b5": 28_000_000,
    "efficientnet-b6": 40_000_000,
    "efficientnet-b7": 63_000_000,
    "mobilenet_v2": 2_000_000,
    "xception": 22_000_000
}