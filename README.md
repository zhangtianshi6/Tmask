# Tmask


## train
python3 train_eval_transoformer.py --dataset data/shapes_train.h5 --encoder large --embedding-dim 64 --num-objects 5 --epochs 500 --name shapes_sub --black_lthr 0.4 --batch-size 70 --layers 2 --seqences-len 10 --choice-model sub


python3 train_eval_transoformer.py --dataset data/cubes_train.h5 --encoder large --embedding-dim 64 --num-objects 5 --epochs 500 --name cubes_sub --black_lthr 0.4 --batch-size 70 --layers 2 --seqences-len 10 --choice-model sub


## test
python3 eval_modules.py --dataset data/shapes_eval.h5 --save-folder checkpoints/shapes_sub --choice-model sub


## test mask
python3 eval_modules_mask.py --dataset data/shapes_eval.h5 --save-folder checkpoints/shapes_sub --choice-model sub

## Tmask
