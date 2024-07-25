@echo off
setlocal

REM Run the first Python script
echo Training ensemble...
python train_ensemble.py --architecture ResUnet --encoder "vgg19" --size 256
IF %ERRORLEVEL% NEQ 0 (
    echo script1.py failed
)

REM Sleep for 10 seconds
timeout /t 600 /nobreak

REM Run the second Python script
echo Training ensemble...
python train_ensemble.py --architecture ResUnet --encoder "inceptionv4" --size 256
IF %ERRORLEVEL% NEQ 0 (
    echo script2.py failed
)

REM Sleep for 10 seconds
timeout /t 600 /nobreak

REM Run the third Python script
echo Training ensemble...
python train_ensemble.py --architecture ResUnet --encoder "xception" --size 256
IF %ERRORLEVEL% NEQ 0 (
    echo script3.py failed
)

REM Sleep for 10 seconds
timeout /t 600 /nobreak


REM Run the fifth Python script
echo Training ensemble...
python train_ensemble.py --architecture EfficientUnetPlusPlus --encoder "efficientnet-b5" --size 256
IF %ERRORLEVEL% NEQ 0 (
    echo script5.py failed
)

REM Sleep for 10 seconds
timeout /t 600 /nobreak

REM Run the sixth Python script
echo Training ensemble...
python train_ensemble.py --architecture EfficientUnetPlusPlus --encoder "efficientnet-b6" --size 256
IF %ERRORLEVEL% NEQ 0 (
    echo script6.py failed
)

echo All scripts completed successfully
endlocal
