# Run training and testing

from argparse import ArgumentParser
from train import train
from test import evaluate_testset

def main(name, model_name, epochs, lr):
    train(name, model_name, epochs, lr)
    evaluate_testset(name, model_name)

if __name__=="__main__": 
    parser = ArgumentParser()
    parser.add_argument("--lr", action="store")
    parser.add_argument("--epochs", action="store")
    parser.add_argument("--name", action="store", required=True)
    parser.add_argument("--model_name", action="store", required=True)
    args = parser.parse_args()
    args = vars(args)

    lr = float(args["lr"]) if args["lr"] else None
    epochs = int(args["epochs"]) if args["epochs"] else None
    name = args["name"]
    model_name = args["model_name"]

    main(name, model_name, epochs, lr)