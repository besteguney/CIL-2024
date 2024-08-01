# CIL-2024

# Prepare data
Download kaggle dataset and save it with the following structure 

```
.
├── ethz-cil-road-segmentation-2024
│   ├── test
│   │	├── images
│   │	|	├── satimage_0.png
│   │	|	└── ... 
│   ├── training
│   │	├── images
│   │	|	├── satimage_0.png
│   │	|	└── ... 
│   │	├── groundtruth
│   │	|	├── satimage_0.png
│   │	|	└── ... 
```


## Scrape data (Optional)
You need to have your own maps api key which you havev to put into collect_data.py (KEY = )

then run:

python .\data\collect_data.py
  

# Train a model
To train a model run:

python .\train_model.py --architecture "architecture" --encoder "encoder" --size size -- n_locs n_locs --model_filename "path/to/model/checkpoint"

Following architectures are supported:
"Unet", "UnetPlusPlus", "FPN", "PSPNet", "PAN", "Linknet", "DeepLabV3Plus", "EfficientUnetPlusPlus", "ResUnet", "ResUnetPlusPlus", "MAnet"

Following encoders are supported:

'resnet18',  'resnet34',  'resnet50',  'resnet101',  'resnet152',  'resnext50_32x4d',  'resnext101_32x8d',  'resnext101_32x16d', 'resnext101_32x32d',  'resnext101_32x48d',  'dpn68',  'dpn68b',  'dpn92',  'dpn98',  'dpn107',  'dpn131',  'vgg11',  'vgg11_bn', 'vgg13',  'vgg13_bn',  'vgg16',  'vgg16_bn',  'vgg19',  'vgg19_bn',  'senet154',  'se_resnet50',  'se_resnet101',  'se_resnet152',  'se_resnext50_32x4d',  'se_resnext101_32x4d',  'densenet121',  'densenet169',  'densenet201',  'densenet161',  'inceptionresnetv2',  'inceptionv4',  'efficientnet-b0',  'efficientnet-b1',  'efficientnet-b2',  'efficientnet-b3',  'efficientnet-b4',  'efficientnet-b5',  'efficientnet-b6',  'efficientnet-b7',  'mobilenet_v2',  'xception'

size: resolutiuon of images to be trained

n_locs: number of scraped locations to use (use 0 if no images were scraped)

model_filename is only used if you want to continue training a model

# Create Submission
To create a submission put all models trained with train_model.py into the folder "saved_models/"

Then run:

python .\infer_ensemble.py "submission_filename.csv"

# Train Roadtracer
run:

python .\roadtracer_main.py

# Roadtracer Inference
run:

python roadtracer_infer.py --load_model 'path/to/roadtracer/checkpoint/file'
