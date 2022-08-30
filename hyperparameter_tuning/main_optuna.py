# This Document was used to do Hyperparameter Tuning.
# To use it one needs to set up a mysql database

import optuna
from sqlalchemy import create_engine
from argparse import ArgumentParser
from train_optuna import train_and_test
from test import evaluate_testset
import matplotlib.pyplot as plt


# Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
def objective(trial, name, model_name, epochs):
    params = {
        "growth_rate": trial.suggest_int("growth_rate", 24, 34),
        "num_init_features": trial.suggest_int("num_init_features", 32, 128),
        "drop_rate": trial.suggest_loguniform("drop_rate", 1e-10, 0.3),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-7, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
    }

    name += f"{trial.number}"

    accuracy = train_and_test(params, trial, name, model_name, epochs)

    return accuracy


if __name__ == "__main__":
    create_study = False
    epochs = 10
    name = "optuna_2_"
    model_name = "cnn_3"

    DB = "mysql://root:1234@localhost:3306/optuna_3"
    study_name = "study_3"

    engine = create_engine(DB)
    if create_study:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            study_name=study_name,
            storage=DB,
        )
    else:
        study = optuna.load_study(study_name=study_name, storage=DB)

    func = lambda trial: objective(trial, name, model_name, epochs)
    study.optimize(func, n_trials=10)

    best_trial = study.best_trial

    print(f"The Best Trial: {best_trial.number}")
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    evaluate_testset(f"{name}{best_trial.number}", model_name)
