## path to train data
export TRAINING_DATA='input/simple_dataset/train_final.pkl'

## model to be used
export MODEL=$1

FOLD=0 python -m src.train
FOLD=1 python -m src.train
FOLD=2 python -m src.train
FOLD=3 python -m src.train
FOLD=4 python -m src.train