import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from dataset import HistopathologicImageDataset
from models.cnn_2 import Net


name = "cnn2"

model_index = 0

# Put this in a DataLoader function
device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#TODO: balance labels?, do image Transformations?
g = torch.Generator()
g.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# Use worker_init_fn() and generator to preserve reproducibility

def load_data():
    dataset = HistopathologicImageDataset(labels_file="./data/train_labels.csv", img_dir="./data/train/", transform=transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float32),
        transforms.Normalize([0.7025, 0.5463, 0.6965], [0.2389, 0.2821, 0.2163]),
    ]))

    train_set, test_set = random_split(dataset, [176_020, 44_005])

    return train_set, test_set

metric_collection = MetricCollection([
    Accuracy(),
    Precision(),
    Recall(),
    F1Score(),
    AUROC()
])


def train(config, checkpoint_dir=None):
    net = Net(config["l1"], config["l2"], img_dim=[3, 96, 96])
    net.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    train_set, test_set = load_data()
    train_loader = DataLoader(dataset=train_set, batch_size=config["batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(dataset=test_set, batch_size=config["batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g)

    losses = []
    accuracy = []
    precision = []
    recall = []
    f1_score = []
    auroc = []

    for epoch in tqdm(range(10)):
        running_loss = 0.0
        list_of_preds = []
        list_of_labels = []

        for idx, data in tqdm(enumerate(train_loader, 0)):
            # get the inputs
            imgs, labels = data[0].to(device), data[1].to(torch.float32).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            preds = net(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            list_of_preds.append(preds)
            list_of_labels.append(labels)

            if idx % 200 == 199: # calculate and print train metrics every 200 mini-batches
                loss = running_loss / 200
                losses.append(loss)

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
            
            # validation loss
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, data in enumerate(test_loader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = criterion(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be passed as the `checkpoint_dir`
            # parameter in future iterations.
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (net.state_dict(), optimizer.state_dict()), path)

            tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

    train_metrics = pd.DataFrame({
        f'accuracy{model_index}': metrics['Accuracy'].item(),
        f'precision{model_index}': metrics['Precision'].item(),
        f'recall{model_index}': metrics['Recall'].item(),
        f'f1_score{model_index}': metrics['F1Score'].item(),
        f'auroc{model_index}': metrics['AUROC'].item()
        })

    train_metrics.to_csv(f'{name}_train.csv')
    print("Finished Training")


def test_best_model(best_trial):
    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    train_set, test_set = load_data()
    test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=False, worker_init_fn=2)

    correct = 0
    total = 0

    list_of_labels = []
    list_of_preds = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            preds = best_trained_model(images)
            list_of_preds.append(preds)
            list_of_labels.append(labels)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    labels = torch.cat(list_of_labels)
    preds = torch.cat(list_of_preds)
    metrics = metric_collection(preds, labels)
    print(metrics)
    test_metrics = pd.DataFrame({
        'accuracy': metrics['Accuracy'].item(),
        'precision': metrics['Precision'].item(),
        'recall': metrics['Recall'].item(),
        'f1_score': metrics['F1Score'].item(),
        'auroc': metrics['AUROC'].item()
        }, index=[0])
    test_metrics.to_csv(f'{name}_test.csv')

    PATH = f"./{name}.pth"
    torch.save(best_trained_model.state_dict(), PATH)
    print("Best trial test set accuracy: {}".format(correct / total))


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.choice([0, 1e-5, 1e-4, 1e-3]),
        "batch_size": tune.choice([8, 16, 32, 64, 128, 256])
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    result = tune.run(
        tune.with_parameters(train),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    #probably unnecessary
    '''
    if ray.util.client.ray.is_connected():
        # If using Ray Client, we want to make sure checkpoint access
        # happens on the server. So we wrap `test_best_model` in a Ray task.
        # We have to make sure it gets executed on the same node that
        # ``tune.run`` is called on.
        from ray.util.ml_utils.node import force_on_current_node
        remote_fn = force_on_current_node(ray.remote(test_best_model))
        ray.get(remote_fn.remote(best_trial))
    else:
    '''
    test_best_model(best_trial)

if __name__=="__main__": 
    #mean, std = calculate_mean_std()
    #print(mean)
    #print(std)
    #name = "cnn2"
    #train(name)
    #evaluate(name)
    main(num_samples=2, max_num_epochs=2, gpus_per_trial=0)