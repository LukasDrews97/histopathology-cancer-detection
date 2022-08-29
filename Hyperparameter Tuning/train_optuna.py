# Instantiates a model, conducts the training and testing for one Trial of the Hyperparameter Tuning Process.

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
import importlib

from data_loading_optuna import get_train_and_test_loader


def train_and_test(param, trial, name, model_name, epochs=None):
    # def train_and_test(name, model_name, epochs=None):
    if epochs == None:
        epochs = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import model, e.g. "cnn_2
    model_module = importlib.import_module(f"architecture.{model_name}")

    # Create neural network
    net = model_module.Net(
        growth_rate=param["growth_rate"],
        num_init_features=param["num_init_features"],
        drop_rate=param["drop_rate"],
    )
    net = net.to(device)

    # Create data loader
    train_loader, test_loader = get_train_and_test_loader(
        batch_size=param["batch_size"]
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        net.parameters(), lr=param["learning_rate"], weight_decay=param["weight_decay"]
    )

    losses = []
    accuracy = 0

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        list_of_preds = []
        list_of_labels = []

        for idx, data in enumerate(train_loader, 0):
            imgs, labels = data[0].to(device), data[1].to(torch.float32).to(device)
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            preds = net(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            list_of_preds.append(preds)
            list_of_labels.append(labels)

            # calculate and print train metrics every 500 mini-batches
            if idx % 500 == 499:
                loss = running_loss / 500
                losses.append(loss)

                print(f"{trial.number}: [{epoch + 1}, {idx + 1:5d}] loss: {loss:.3f}")
                running_loss = 0.0
                list_of_labels.clear()
                list_of_preds.clear()

        # test
        metric_collection_test = MetricCollection(
            [Accuracy(), Precision(), Recall(), F1Score()]
        )
        metric_collection_test = metric_collection_test.to(device)

        list_of_labels = []
        list_of_preds = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                imgs, labels = data
                imgs, labels = imgs.to(device), labels.to(device)
                preds = net(imgs)
                list_of_preds.append(preds)
                list_of_labels.append(labels)

        labels = torch.cat(list_of_labels)
        preds = torch.cat(list_of_preds)
        labels, preds = labels.to(device), preds.to(device)

        # Calculate metrics
        metrics = metric_collection_test(preds, labels)

        # Create dataframe
        test_metrics = pd.DataFrame(
            {
                "accuracy": metrics["Accuracy"].item(),
                "precision": metrics["Precision"].item(),
                "recall": metrics["Recall"].item(),
                "f1_score": metrics["F1Score"].item(),
            },
            index=[0],
        )

        # Save testing metrics
        test_metrics.to_csv(f"{name}_test.csv")

        # Add prune mechanism
        accuracy = metrics["Accuracy"].item()
        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # save model parameters
    PATH = f"./{name}.pth"
    torch.save(net.state_dict(), PATH)

    return accuracy
