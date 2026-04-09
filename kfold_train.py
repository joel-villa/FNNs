import torch
import time
import numpy as np
from load_data import npz_load
import onnxruntime as ort
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from baseline import prepare_data, make_fnn

GPU = True
INPUT_DIMENSION = 8000
# Binary classification -> one neuron in output layer
OUTPUT_NEURONS = 1

def get_device():
    if GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_fnn_kfold(num_iter, num_hidden_layers, hidden_neurons, learning_rate, weight_decay, batch_size=32, kfolds=5, save_path=None):
    device = get_device()
    # Get data
    x_train, y_train, x_test, y_test = npz_load(INPUT_DIMENSION)
    x_train, y_train = prepare_data(x_train, y_train)
    x_test, y_test = prepare_data(x_test, y_test)

    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # make kfold
    kfold = KFold(n_splits=kfolds, shuffle=True)
    L = torch.nn.BCEWithLogitsLoss()
    results = {}

    fold_train_accs = []
    fold_val_accs = []
    fold_test_accs = []
    fold_times = []

    nets = []


    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=val_subsampler)

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
        test_acc = get_binary_accuracy(net, test_loader, device)

        fold_train_accs.append(train_acc)
        fold_val_accs.append(val_acc)
        fold_test_accs.append(test_acc)
        fold_times.append(end - start)


        print(f"fold {fold + 1}/{kfolds}: "
              f"train_acc = {train_acc:.2f}%, "
              f"val_acc = {val_acc:.2f}%, "
              f"test_acc = {test_acc:.2f}%, "
              f"time = {end - start:.4f}s")

        nets.append(net)

    # full eval across all fold nets
    for net in nets:
        net.eval()

    correct, total = 0, 0

    computed_res = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            prob_per_model = []
            for net in nets:
                output = net.forward(xb)
                prob = (torch.sigmoid(output))
                prob_per_model.append(prob)
            
            mean_prob = torch.stack(prob_per_model, dim=0).mean(dim=0)
            predicted = (mean_prob >= 0.5).float()
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    
        computed_res = 100.0 * correct / total

    results = {
        "avg_train_acc": np.mean(fold_train_accs),
        "avg_val_acc": np.mean(fold_val_accs),
        "avg_test_acc": np.mean(fold_test_accs),
        "avg_time": np.mean(fold_times),
        "kfolds": kfolds,
        "fold_train_accs": fold_train_accs,
        "fold_val_accs": fold_val_accs,
        "fold_times": fold_times,
        "full_test_acc": computed_res
    }

    print()
    print(f"Results:")
    print(f"average train accuracy: {results['avg_train_acc']:.2f}%")
    print(f"average validation accuracy: {results['avg_val_acc']:.2f}%")
    print(f"average test accuracy: {results['avg_test_acc']:.2f}%")
    print(f"average training time: {results['avg_time']:.4f}s")
    print(f"Full evaluation all nets : {results['full_test_acc']:.4f}%")

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
        "avg_test_acc": 0.0,
        "avg_time": 0.0,
        "kfolds": 0,
        "full_test_acc": 0.0
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

        if results["full_test_acc"] > best["full_test_acc"]:
            best = results

    return best, all_results

# no k_fold
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
    print("No kfold model results:")
    print(f"train_acc = {train_acc:.2f}%")
    print(f"test_acc = {test_acc:.2f}%")
    print(f"train_time = {end - start:.4f}s")

    return results

if __name__ == "__main__":

    print("Starting!")

    device = get_device()
    print(f"We are using: {device}")

    # existing params
    num_iter = 32
    hidden_neurons = 64
    num_hidden_layers = 3
    learning_rate = 1e-5
    weight_decay = 1e-7

    k_values = [2]

    best, all_results = test_kfold_values(
        num_iter=num_iter,
        num_hidden_layers=num_hidden_layers,
        hidden_neurons=hidden_neurons,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        k_values=k_values
    )

    print("All k-fold results across ks:")
    for result in all_results:
        print(result)

    print()

    print("Best k-fold result:")
    print(best)
    print()

    final_results = traintest_single_fnn(
        num_iter=num_iter,
        num_hidden_layers=num_hidden_layers,
        hidden_neurons=hidden_neurons,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    print("Without k_fold:")
    print(final_results)
