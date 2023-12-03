:: call training function
:: NOTE: very slow on cpus... probably need to get this running on a cluster

:: set vars
set DATA_DIR=l:\10_IO\2307_super\ins\SRCNN\set5
set OUT_DIR=l:\10_IO\2307_super\outs\SRCNN\play



:: activate environment
cd ..
call env\conda_activate.bat

::  call training
python train.py --train-file "%DATA_DIR%/91-image_x3.h5" --eval-file "%DATA_DIR%/Set5_x3.h5" --outputs-dir "%OUT_DIR%" --scale 3 --lr 1e-4 --batch-size 16 --num-epochs 10 --num-workers 8  --seed 123 

ECHO finished
cmd.exe /k