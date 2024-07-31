@echo off
setlocal


REM Run the first Python script
echo Training ensemble...
python train_ensemble.py --architecture EfficientUnetPlusPlus --encoder "efficientnet-b5" --size 256
IF %ERRORLEVEL% NEQ 0 (
    echo script1.py failed
)

REM Sleep for 10 seconds
timeout /t 600 /nobreak

REM Run the fifth Python script
echo Training ensemble...
python train_ensemble.py --architecture Unet --encoder "vgg19" --size 384
IF %ERRORLEVEL% NEQ 0 (
    echo script5.py failed
)

echo All scripts completed successfully
endlocal
