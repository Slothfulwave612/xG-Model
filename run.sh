## model to be used
export TYPE=$1
export MODEL=$2
export SCALE_TYPE=$3
export DATA=$4

if [ $TYPE = "basic" ]       ## for basic model
then

    if [ $DATA = "True" ]
    then
        python -m src.dataset                       ## create dataset
    fi

    echo "Creating train-test data..."

    python -m src.train_test_data               ## create train and test data

    if [ $MODEL = "log_regg" ]
    then
        FILE="ohe"
        ENC_TYPE="ohe"
    else
        FILE="label"
        ENC_TYPE="label"
    fi

    echo "Categorical Encoding..."

    python -m src.categorical $ENC_TYPE         ## categorical encoding

    echo "Preparing dataset for training and prediction..."

    python -m src.prepare_dataset $ENC_TYPE     ## prepare dataset for training and prediction

    ## path to the required files
    export TRAINING_DATA="input/basic_dataset/train_test_data_final/train_${FILE}_final.pkl"
    export TEST_DATA="input/basic_dataset/train_test_data_final/test_${FILE}_final.pkl"
    export SAVE_PATH="input/basic_dataset/train_test_data_result"

    python -m src.train                         ## train, predict and save
fi