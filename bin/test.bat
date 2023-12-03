:: call training function
:: NOTE: very slow on cpus... probably need to get this running on a cluster

:: set vars
set DATA_DIR=l:\10_IO\2307_super\ins\SRCNN\set5
set OUT_DIR=l:\10_IO\2307_super\outs\SRCNN\run

::trained model weights
set WEIGHTS_FP=l:\10_IO\2307_super\outs\SRCNN\weights\srcnn_x4.pth

:: raw image to downscale
set IMAGE_FP=l:\10_IO\2307_super\outs\SRCNN\set5_x3\lr_4.png


:: activate environment
cd ..
call env\conda_activate.bat

::  call training
python test.py --weights-file %WEIGHTS_FP% --image-file %IMAGE_FP% --scale 4 --outdir %OUT_DIR%

ECHO finished
cmd.exe /k