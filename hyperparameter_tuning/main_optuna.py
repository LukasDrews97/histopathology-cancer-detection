# This Document was used to do Hyperparameter Tuning.
# To use it one needs to set up a mysql database

import optuna
from train_optuna import train_and_test
#import matplotlib.pyplot as plt
import pandas as pd
import joblib
from importlib.machinery import SourceFileLoader
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

test_module = SourceFileLoader("test", f"{parent}/test.py").load_module()


# Define a set of hyperparameter values
def sample_densenet_params(trial):
    return {
        "growth_rate": trial.suggest_int("growth_rate", 24, 34),
        "num_init_features": trial.suggest_int("num_init_features", 32, 128),
        "drop_rate": trial.suggest_loguniform("drop_rate", 1e-10, 0.3),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-7, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
    }


# Load hyperparamter values, build the model, train the model, and evaluate the accuracy
def objective(trial, name, model_name, epochs):
    params = sample_densenet_params(trial)
    name += f"{trial.number}"
    accuracy = train_and_test(params, trial, name, model_name, epochs)
    return accuracy


if __name__ == "__main__":
    # Create study or continue an old one
    create_study = True
    # Number of epochs to train on
    epochs = 10
    # Name of the output files
    name = "optuna"
    # Name of the model architecture, e.g. densenet
    model_name = "densenet"
    # Name of the study file
    study_name = "study"

    # Create study, maximizing accuracy
    if create_study:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            study_name=study_name,
        )
    else:
        study = joblib.load(f"{study_name}.pkl")

    # Maximize accuracy is the training objective
    func = lambda trial: objective(trial, name, model_name, epochs)
    study.optimize(func, n_trials=10, show_progress_bar=True)

    # Save best model
    best_trial = study.best_trial

    # save study and data
    df = study.trials_dataframe()
    joblib.dump(study, f'{study_name}.pkl')
    df.to_csv(f'{study_name}_tuning_data.csv')

    
    print(f"The Best Trial: {best_trial.number}")
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    params = {}
    params["growth_rate"] = best_trial.params["growth_rate"]
    params["num_init_features"] = best_trial.params["num_init_features"]
    params["drop_rate"] = best_trial.params["drop_rate"]

    test_module.evaluate_testset(name=f"{name}{best_trial.number}", model_name=model_name, params=params, labels_file="../data/train_labels.csv", img_dir="../data/train/")