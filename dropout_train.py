import torch
import time
import numpy as np
from load_data import npz_load
import onnxruntime as ort
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

from baseline import prepare_data

GPU = True
INPUT_DIMENSION = 8000
# Binary classification -> one neuron in output layer
OUTPUT_NEURONS = 1

def get_device():
    if GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def make_fnn_dropout(h_neurons, num_hidden_layers, dropout_probability=0.5):
    layers = []

    # First layer
    layers.append(torch.nn.Linear(INPUT_DIMENSION, h_neurons))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Dropout(p=dropout_probability))

    # Hidden layers
    for _ in range(num_hidden_layers - 1):
        layers.append(torch.nn.Linear(h_neurons, h_neurons))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(p=dropout_probability))

    # Output layer
    layers.append(torch.nn.Linear(h_neurons, OUTPUT_NEURONS))

    # * in python -> convert list to function
    return torch.nn.Sequential(*layers)

def train_fnn_dropout(num_iter, num_hidden_layers, 
                      hidden_neurons, learning_rate=0.001, 
                      weight_decay=0, batch_size=32, save_path=None, 
                      dropout_probability=0.5):
    device = get_device()
    net = make_fnn_dropout(h_neurons=hidden_neurons, num_hidden_layers=num_hidden_layers, dropout_probability=dropout_probability).to(device)

    # Get data
    x_train, y_train, x_test, y_test = npz_load(INPUT_DIMENSION)
    x_train, y_train = prepare_data(x_train, y_train)
    
    optimizer = torch.optim.Adam(net.parameters(),
                                lr=learning_rate, 
                                weight_decay=weight_decay)
    
    # loss fn
    L = torch.nn.BCEWithLogitsLoss()

    x_tensor = torch.from_numpy(x_train).float()  
    y_tensor = torch.from_numpy(y_train).float()

    train_dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    start = time.perf_counter()

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

    # do testing
    x_test, y_test = prepare_data(x_test, y_test)

    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_acc = get_binary_accuracy(net=net, data_loader=train_loader, device=device)
    test_acc = get_binary_accuracy(net=net, data_loader=test_loader, device=device)

    results = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_time": end - start
    }

    return results

def train_fnn_dropout_bagging(num_iter, num_hidden_layers, 
                              hidden_neurons, learning_rate=0.001, 
                              weight_decay=0, batch_size=32, save_path=None, 
                              dropout_probability=0.5, num_models=8):
    device = get_device()

    bag_nets = []

    x_train, y_train, x_test, y_test = npz_load(INPUT_DIMENSION)
    x_train, y_train = prepare_data(x_train, y_train)

    n_samples = len(x_train)

    start = time.perf_counter()

    for i in range(num_models):
        # sample dataset
        rand_idxs = np.random.choice(n_samples, size=n_samples, replace=True)

        x_subsample = x_train[rand_idxs]
        y_subsample = y_train[rand_idxs]

        net = make_fnn_dropout(h_neurons=hidden_neurons, num_hidden_layers=num_hidden_layers, dropout_probability=dropout_probability).to(device)

        optimizer = torch.optim.Adam(net.parameters(),
                                lr=learning_rate, 
                                weight_decay=weight_decay)
        
        L = torch.nn.BCEWithLogitsLoss()

        # create dataset from sample
        x_tensor = torch.from_numpy(x_subsample).float()
        y_tensor = torch.from_numpy(y_subsample).float()

        train_dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # train nn model
        net.train()
        for epoch in range(num_iter):
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                outputs = net.forward(xb)
                loss = L(outputs, yb)
                loss.backward()
                optimizer.step()
        
        bag_nets.append(net)
    
    end = time.perf_counter()

    # do testing
    x_test, y_test = prepare_data(x_test, y_test)

    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_acc = predict_bagged_fnn(bag_nets=bag_nets, data_loader=train_loader, device=device)
    test_acc = predict_bagged_fnn(bag_nets=bag_nets, data_loader=test_loader, device=device)

    results = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_time": end - start
    }

    return results

# for use with a single network
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

def predict_bagged_fnn(bag_nets, data_loader, device):
    # set all nets to eval mode before proceeding with prediction
    for net in bag_nets:
        net.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            prob_per_model = []
            for net in bag_nets:
                output = net.forward(xb)
                prob = (torch.sigmoid(output))
                prob_per_model.append(prob)
            
            mean_prob = torch.stack(prob_per_model, dim=0).mean(dim=0)
            predicted = (mean_prob >= 0.5).float()
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    
    return 100.0 * correct / total
    

if __name__ == "__main__":
    # seed = 10
    # np.random.seed(seed)

    print("Starting!")

    device = get_device()
    print(f"We are using: {device}")

    # existing params
    num_iter = 16
    hidden_neurons = 64
    num_hidden_layers = 3
    learning_rate = 1e-5
    weight_decay = 1e-7

    num_models = 8

    batch_size = 32

    dropout_probabilities = [0.2, 0.3, 0.4, 0.5]

    best_results_bag = {
        "train_acc": 0.0,
        "test_acc": 0.0,
        "train_time": 99999.0
    }

    best_results_drop = {
        "train_acc": 0.0,
        "test_acc": 0.0,
        "train_time": 99999.0
    }

    print("Proceeding to loop over dropout probabilities...")

    for dropout_probability in dropout_probabilities:
        print(f"Training Bag Net with probability: {dropout_probability}")
        results = train_fnn_dropout_bagging(num_iter=num_iter, 
                        num_hidden_layers=num_hidden_layers, 
                        hidden_neurons=hidden_neurons,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        batch_size=batch_size,
                        dropout_probability=dropout_probability,
                        num_models=num_models)

        curr_train_acc_bag = results["train_acc"]
        curr_test_acc_bag = results["test_acc"]
        curr_train_time_bag = results["train_time"]

        print(f"Bag Net with Dropout Prob: {dropout_probability}")
        print(f"    train_acc = {curr_train_acc_bag:.2f}%")
        print(f"    test_acc = {curr_test_acc_bag:.2f}%")
        print(f"    train_time = {curr_train_time_bag:.2f}s")

        print()

        # we want best test accuracy
        if curr_test_acc_bag > best_results_bag["test_acc"]:
            best_results_bag = {
                "train_acc": curr_train_acc_bag,
                "test_acc": curr_test_acc_bag,
                "train_time": curr_train_time_bag,
                "dropout_probability": dropout_probability
            }

    for dropout_probability in dropout_probabilities:
        print(f"Training Drop Net with probability: {dropout_probability}")
        results = train_fnn_dropout(num_iter=num_iter, 
                        num_hidden_layers=num_hidden_layers, 
                        hidden_neurons=hidden_neurons,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        batch_size=batch_size,
                        dropout_probability=dropout_probability,
                        num_models=num_models)

        curr_train_acc_drop = results["train_acc"]
        curr_test_acc_drop = results["test_acc"]
        curr_train_time_drop = results["train_time"]

        print(f"Drop Net with Dropout Prob: {dropout_probability}")
        print(f"    train_acc = {curr_train_acc_drop:.2f}%")
        print(f"    test_acc = {curr_test_acc_drop:.2f}%")
        print(f"    train_time = {curr_train_time_drop:.2f}s")

        print()

        # we want best test accuracy
        if curr_test_acc_drop > best_results_drop["test_acc"]:
            best_results_drop = {
                "train_acc": curr_train_acc_drop,
                "test_acc": curr_test_acc_drop,
                "train_time": curr_train_time_drop,
                "dropout_probability": dropout_probability
            }

    print(f"Best Bag Net performance:")
    print(f"    num_iter: {num_iter}")
    print(f"    hidden_neurons: {hidden_neurons}")
    print(f"    num_hidden_layers: {num_hidden_layers}")
    print(f"    learning_rate: {learning_rate}")
    print(f"    weight_decay: {weight_decay}")
    print(f"    batch_size: {batch_size}")
    print(f"    num_models: {num_models}")
    print(f"    dropout_probability: {best_results_bag['dropout_probability']}")
    print(f"    train_accuracy: {best_results_bag['train_acc']}%")
    print(f"    test_accuracy: {best_results_bag['test_acc']}%")
    print(f"    train_time: {best_results_bag['train_time']}s")
    print()

    print(f"Best Drop Net performance:")
    print(f"    num_iter: {num_iter}")
    print(f"    hidden_neurons: {hidden_neurons}")
    print(f"    num_hidden_layers: {num_hidden_layers}")
    print(f"    learning_rate: {learning_rate}")
    print(f"    weight_decay: {weight_decay}")
    print(f"    batch_size: {batch_size}")
    print(f"    num_models: {num_models}")
    print(f"    dropout_probability: {best_results_drop['dropout_probability']}")
    print(f"    train_accuracy: {best_results_drop['train_acc']}%")
    print(f"    test_accuracy: {best_results_drop['test_acc']}%")
    print(f"    train_time: {best_results_drop['train_time']}s")
    print()



    print("Ending!")





        



