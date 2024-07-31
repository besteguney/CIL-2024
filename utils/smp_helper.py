
import segmentation_models_pytorch as smp

def model_init(encoder_name, architecture="Unet"):
    if architecture == "Unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            classes=1
        )
    elif architecture == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            classes=1
        )
    elif architecture == "FPN":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            in_channels=3,
            classes=1,
            upsampling=4
        )
    elif architecture == "PSPNet":
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            psp_out_channels=512,
            psp_use_batchnorm=True,
            psp_dropout=0.2,
            in_channels=3,
            classes=1,
            upsampling=8
        )
    elif architecture == "PAN":
        model = smp.PAN(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            decoder_channels=32,
            in_channels=3,
            classes=1
        )
    elif architecture == "Linknet":
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            in_channels=3,
            classes=1
        )
    elif architecture == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            in_channels=3,
            classes=1,
            upsampling=4
        )
    elif architecture == "EfficientUnetPlusPlus":
        model = smp.EfficientUnetPlusPlus(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            classes=1
        )
    elif architecture == "ResUnet":
        model = smp.ResUnet(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            classes=1
        )
    elif architecture == "ResUnetPlusPlus":
        model = smp.ResUnetPlusPlus(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            classes=1
        )
    elif architecture == "MAnet":
        model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_pab_channels=64,
            in_channels=3,
            classes=1
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model