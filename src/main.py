import argparse
from features import load_and_preprocess
from models import predict_toxicity
from sklearn import set_config

set_config(transform_output = "pandas")
TRAINING_DATA_PATH =""
FITTED_MODEL_PATH = "model.joblib"
TOX_TARGET_PATH ="predicted_toxicity.csv"
COLUMNS_TO_REMOVE_PATH = "columns_to_remove_in_preprocessing.json"
def prepare_data():
    # convert sdf  file to rdkit
    # calculate the desired descriptors
    # 
    pass
def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict pLC50 for molecules loaded from an SDF file.")
    parser.add_argument("filename", help="Path to the SDF file containing molecular structures.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    pre_processed_df = load_and_preprocess(args.filename,COLUMNS_TO_REMOVE_PATH)
    predicted_toxicity = predict_toxicity(pre_processed_df,FITTED_MODEL_PATH)
    predicted_toxicity.insert(0, "compound_id", predicted_toxicity.index.to_list())
    predicted_toxicity.to_csv(TOX_TARGET_PATH)
if __name__=="__main__":
    #pre_processed_df = load_and_preprocess("data/chin-qspr-dataset.sdf",COLUMNS_TO_REMOVE_PATH)
    #predicted_toxicity = predict_toxicity(pre_processed_df,FITTED_MODEL_PATH)
    #predicted_toxicity.to_csv(TOX_TARGET_PATH)
    main()