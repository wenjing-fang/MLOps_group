#!python3

##add the path to the folder mlops project to identify the package or is there other method to adress this issue?
import sys
sys.path.append('D:/MainWorkingPlace/MLOPS/final project/MLOps_group')
####

import argparse
from html import parser
import logging 
import os

import project_code.functions as f
import project_code.models as models 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S,",
)

def create_parser():
    parser = argparse.ArgumentParser(description="MLOPS CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List models command (it has no additional parameters)
    subparsers.add_parser("list", help="List available models")

    # Predict command (it requires a model name, indicate the train or eval mode)
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--model", help="Model name to use", required=True)
    predict_parser.add_argument("--mode", help="Choose between train or eval mode", required=True)

    return parser

def get_script_args():
    parser=create_parser()
    return parser.parse_args()

def predict(model_name,mode):
    # input data
    raw_data = f.read_dataset()
    data = f.preprocess_data(raw_data)
    X_train, X_test, y_train, y_test = f.split_data(data)
    model = models.get_model(model_name)

    if mode=='train':
        model.fit(X_train, y_train)
        f.save_model(model, model_name)
        print(f'the model {model_name} has been successfully saved in folder checkpoints')
    elif mode=='eval':
        loaded_model = f.load_model(model_name)
        metrics = f.evaluate_model(loaded_model, X_test, y_test)  
        print("Model: ", model_name)
        print(f"F1 Score: {metrics[0]},Precision: {metrics[1]}")

def get_model_names():
        filenames=os.listdir('./checkpoints')
        modelnames=[f[:f.index('.')] for f in filenames]
        return modelnames



def main():
    args=get_script_args()
    if args.command == 'list':
        for model in get_model_names():
            print(f"- {model}")
    if args.command =='predict':
        predict(args.model,args.mode)

if __name__=='__main__':
    main()
        

