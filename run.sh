## path to train data
export TRAINING_DATA='input/simple_dataset/train_test_data_final/train_ohe_final.pkl'
export TEST_DATA='input/simple_dataset/train_test_data_final/test_ohe_final.pkl'
export SAVE_PATH='input/simple_dataset/train_test_data_result'

## model to be used
export MODEL=$1

python -m src.train