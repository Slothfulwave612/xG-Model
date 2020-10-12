## model to be used
export TYPE=$1
export MODEL=$2
export SCALE_TYPE=$3
export DATA=$4
export ENCODE=$5

if [ $DATA = "True" ]
then
    python -m src.dataset $TYPE                   ## create dataset
    
    echo "Creating train-test data..."
    python -m src.train_test_data $TYPE               ## create train and test data
fi

if [ $MODEL = "log_regg" ]
then
    FILE="ohe"
    ENC_TYPE="ohe"
else
    FILE="label"
    ENC_TYPE="label"
fi

if [ $ENCODE = "True" ]
then
    echo "Categorical Encoding..."
    python -m src.categorical $TYPE $ENC_TYPE         ## categorical encoding

    echo "Preparing dataset for training and prediction..."
    python -m src.prepare_dataset $TYPE $ENC_TYPE     ## prepare dataset for training and prediction
fi

## path to the required files
export TRAINING_DATA="input/${TYPE}_dataset/train_test_data_final/train_${FILE}_final.pkl"
export TEST_DATA="input/${TYPE}_dataset/train_test_data_final/test_${FILE}_final.pkl"
export SAVE_PATH="input/${TYPE}_dataset/train_test_data_result"

echo "Model building..."

python -m src.train                               ## train, predict and save
