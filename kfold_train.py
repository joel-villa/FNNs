import torch
import time
import numpy as np
from load_data import npz_load
import onnxruntime as ort
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from baseline import prepare_data, make_fnn

GPU = False

# 5,000 features -> first layer has 5000 input neurons
INPUT_DIMENSION = 5000

# Binary classification -> one neuron in output layer
OUTPUT_NEURONS = 1

def get_device():
    if GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_fnn_kfold(num_iter, num_hidden_layers, hidden_neurons, learning_rate, weight_decay, batch_size=10, kfolds=5, save_path=None):
    device = get_device()
    # Get data
    x_train, y_train, _, _ = npz_load()
    x_train, y_train = prepare_data(x_train, y_train)

    # convert data to tensors, where x_tensor is (N, input dimension) and y is (N, 1)
    x_tensor = torch.from_numpy(x_train).float()
    y_tensor = torch.from_numpy(y_train).float()

    dataset = TensorDataset(x_tensor, y_tensor)

    # make kfold
    kfold = KFold(n_splits=kfolds, shuffle=True)
    L = torch.nn.BCEWithLogitsLoss()
    results = {}

    fold_train_accs = []
    fold_val_accs = []
    fold_times = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=val_subsampler)

        # make net
        net = make_fnn(h_neurons=hidden_neurons, num_hidden_layers=num_hidden_layers).to(device)
        # set optimizer
        optimizer = torch.optim.Adam(net.parameters(),
                            lr=learning_rate, 
                            weight_decay=weight_decay)
        
        start = time.perf_counter() #Timing training

        # do training
        for epoch in range(0, num_iter):
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                outputs = net.forward(xb)
                loss = L(outputs, yb)
                loss.backward()
                optimizer.step()

        end = time.perf_counter()

        train_acc = get_binary_accuracy(net, train_loader, device)
        val_acc = get_binary_accuracy(net, val_loader, device)

        fold_train_accs.append(train_acc)
        fold_val_accs.append(val_acc)
        fold_times.append(end - start)

        print(f"fold {fold + 1}/{kfolds}: "
              f"train_acc = {train_acc:.2f}%, "
              f"val_acc = {val_acc:.2f}%, "
              f"time = {end - start:.4f}s")

    results = {
        "avg_train_acc": np.mean(fold_train_accs),
        "avg_val_acc": np.mean(fold_val_accs),
        "avg_time": np.mean(fold_times),
        "kfolds": kfolds,
        "fold_train_accs": fold_train_accs,
        "fold_val_accs": fold_val_accs,
        "fold_times": fold_times,
    }

    print()
    print(f"{kfolds}-fold results:")
    print(f"average train accuracy: {results['avg_train_acc']:.2f}%")
    print(f"average validation accuracy: {results['avg_val_acc']:.2f}%")
    print(f"average training time: {results['avg_time']:.4f}s")

    return results

def get_binary_accuracy(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            output = net(xb)
            predicted = (torch.sigmoid(output) >= 0.5).float()
            total += yb.size(0)
            correct += (predicted == yb).sum().item()

    return 100.0 * correct / total

def test_kfold_values(num_iter, num_hidden_layers, hidden_neurons, learning_rate, weight_decay, k_values, batch_size=32):
    best = {
        "avg_val_acc": 0.0,
        "avg_train_acc": 0.0,
        "avg_time": 0.0,
        "kfolds": 0
    }

    all_results = []

    for k in k_values:
        print(f"\nTesting k = {k}")
        results = train_fnn_kfold(num_iter=num_iter,
                                  num_hidden_layers=num_hidden_layers,
                                  hidden_neurons=hidden_neurons,
                                  learning_rate=learning_rate,
                                  weight_decay=weight_decay,
                                  batch_size=batch_size,
                                  kfolds=k)

        all_results.append(results)

        if results["avg_val_acc"] > best["avg_val_acc"]:
            best = results

    return best, all_results

def traintest_single_fnn(num_iter, num_hidden_layers, hidden_neurons, learning_rate, weight_decay, batch_size=32):
    device = get_device()

    # Get data
    x_train, y_train, x_test, y_test = npz_load()
    x_train, y_train = prepare_data(x_train, y_train)
    x_test, y_test = prepare_data(x_test, y_test)

    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    net = make_fnn(h_neurons=hidden_neurons, num_hidden_layers=num_hidden_layers).to(device)

    # set optimizer
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    L = torch.nn.BCEWithLogitsLoss()

    start = time.perf_counter()

    # do training
    for epoch in range(0, num_iter):
        current_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = net.forward(xb)
            loss = L(outputs, yb)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

    end = time.perf_counter()

    train_acc = get_binary_accuracy(net, train_loader, device)
    test_acc = get_binary_accuracy(net, test_loader, device)

    results = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_time": end - start
    }

    print()
    print("Final model results:")
    print(f"train_acc = {train_acc:.2f}%")
    print(f"test_acc = {test_acc:.2f}%")
    print(f"train_time = {end - start:.4f}s")

    return results

if __name__ == "__main__":

    print("Starting!")

    device = get_device()
    print(f"We are using: {device}")

    # existing params
    num_iter = 2
    hidden_neurons = 1024
    num_hidden_layers = 4
    learning_rate = 1e-5
    weight_decay = 0.001

    k_values = [2]

    best, all_results = test_kfold_values(
        num_iter=num_iter,
        num_hidden_layers=num_hidden_layers,
        hidden_neurons=hidden_neurons,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        k_values=k_values
    )

    print("All k-fold results:")
    for result in all_results:
        print(result)

    print()

    print("Best k-fold result:")
    print(best)

    final_results = traintest_single_fnn(
        num_iter=num_iter,
        num_hidden_layers=num_hidden_layers,
        hidden_neurons=hidden_neurons,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    print(final_results)
