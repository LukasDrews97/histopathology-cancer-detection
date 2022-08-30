# This Document was used to do Hyperparameter Tuning.
# To use it one needs to set up a mysql database

import optuna
from train_optuna_2 import train_and_test
#import matplotlib.pyplot as plt
#import pandas as pd
import joblib
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import test



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
    create_study = True
    epochs = 10
    name = "optuna"
    model_name = "densenet"
    study_name = "study"

    if create_study:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            study_name=study_name,
        )
    else:
        study = joblib.load(f"{study_name}.pkl")

    func = lambda trial: objective(trial, name, model_name, epochs)
    study.optimize(func, n_trials=10, show_progress_bar=True)

    best_trial = study.best_trial

    # save study and data
    df = study.trials_dataframe()
    joblib.dump(study, f'{study_name}.pkl')
    df.to_csv(f'{study_name}_tuning_data.csv')


    print(f"The Best Trial: {best_trial.number}")
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    test.evaluate_testset(f"{name}{best_trial.number}", model_name)
