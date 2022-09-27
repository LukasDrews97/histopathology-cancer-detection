# Instantiates a model, conducts the training and saves the model

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
import importlib

from data_loading import get_train_and_test_loader

def train(name, model_name, epochs=None, lr=None):
    """
    Train a deep learning model.
    Args:
        name:
            Name of the output files.
        model_name:
            Module name of the model architecture.
        epochs:
            Number of epochs the model should train. Default is 10.
        lr:
            Learning rate for the model training. Default is 0.001 .
    """
    if epochs == None:
        epochs = 10

    if lr == None:
        lr = 1e-3
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import model, e.g. "cnn_2
    model_module = importlib.import_module(f"architecture.{model_name}")

    # Create neural network
    net = model_module.Net(img_dim=[3, 96, 96])
    net.to(device)

    # Create data loader
    train_loader, _ = get_train_and_test_loader()

    # Create training metrics
    metric_collection = MetricCollection([
        Accuracy(),
        Precision(),
        Recall(),
        F1Score(),
        AUROC()
    ])

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    losses = []
    accuracy = []
    precision = []
    recall = []
    f1_score = []
    auroc = []

    # Training loop
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        list_of_preds = []
        list_of_labels = []

        for idx, data in tqdm(enumerate(train_loader, 0)):
            imgs, labels = data[0].to(device), data[1].to(torch.float32).to(device)

            # make predictions, calculate loss and backpropagate
            optimizer.zero_grad()
            preds = net(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            list_of_preds.append(preds)
            list_of_labels.append(labels)

            # calculate and print train metrics every 200 mini-batches
            if idx % 200 == 199:
                loss = running_loss / 200
                losses.append(loss)

                # Calculate metrics
                metrics = metric_collection(torch.cat(list_of_preds).cpu(), torch.cat(list_of_labels).cpu().to(torch.int8))
                accuracy.append(metrics["Accuracy"].item())
                precision.append(metrics["Precision"].item())
                recall.append(metrics["Recall"].item())
                f1_score.append(metrics["F1Score"].item())
                auroc.append(metrics["AUROC"].item())

                print(f'[{epoch + 1}, {idx + 1:5d}] loss: {loss:.3f}')
                print(f'[{epoch + 1}, {idx + 1:5d}] accuracy: {accuracy[-1]:.3f} precision: {precision[-1]:.3f} recall: {recall[-1]:.3f} f1_score: {f1_score[-1]:.3f} auroc: {auroc[-1]:.3f}')
                running_loss = 0.0
                list_of_labels.clear()
                list_of_preds.clear()

    # Create dataframe
    train_metrics = pd.DataFrame(
        {'bce_loss': losses, 
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall,
        'f1_score': f1_score,
        'auroc': auroc
        })

    # save training metrics
    train_metrics.to_csv(f'{name}_train.csv')

    # save model parameters
    PATH = f"./{name}.pth"
    torch.save(net.state_dict(), PATH)




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

    train(name, model_name, epochs, lr)