import torch
import time
import numpy as np
from load_data import npz_load
import onnxruntime as ort
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from baseline import make_fnn, prepare_data

# 5,000 features -> first layer has 5000 input neurons
INPUT_DIMENSION = 5000

# Binary classification -> one neuron in output layer
OUTPUT_NEURONS = 1

def get_path(hlayers, hneurons, lr, wd, iter):
    return f"kfold_models/baseline_{hlayers}_hlayers_{hneurons}_hneurons_{lr}_{wd}_{iter}_fold.onnx"

def train_fnn_kfold(num_iter, num_hidden_layers, hidden_neurons, learning_rate, weight_decay, batch_size=10, k_folds=5, save_path=None):
    # some GPU stuff I was testing
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = make_fnn(h_neurons=hidden_neurons, num_hidden_layers=num_hidden_layers).to(device)

    # Get data
    x_train, y_train, _, _ = npz_load()
    x_train, y_train = prepare_data(x_train, y_train)

    # convert data to tensors, where x_tensor is (N, input dimension) and y is (N, 1)
    x_tensor = torch.from_numpy(x_train).float()
    y_tensor = torch.from_numpy(y_train).float()

    dataset = TensorDataset(x_tensor, y_tensor)

    # make kfold
    kfold = KFold(n_splits=k_folds, shuffle=True)
    L = torch.nn.BCEWithLogitsLoss()
    results = {}


    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_subsampler)

        # make net
        net = make_fnn(h_neurons=hidden_neurons, num_hidden_layers=num_hidden_layers)
        # set optimizer
        optimizer = torch.optim.Adam(net.parameters(),
                            lr=learning_rate, 
                            weight_decay=weight_decay)
        
        start = time.perf_counter() #Timing training

        # do training
        for epoch in range(0, num_iter):
            current_loss = 0.0

            for data in train_loader:
                xb, yb = data
                optimizer.zero_grad()
                outputs = net.forward(xb)
                loss = L(outputs, yb)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()

        end = time.perf_counter()
        print(f"Train time: {end - start:.6f} seconds")

        # validation step, evaluation
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                output = net(xb)
                predicted = (torch.sigmoid(output) >= 0.5).float()  # binary classification
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
            
#        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        results[fold] = 100.0 * (correct / total)

        # save
        if save_path:
            tensor_x = torch.rand((1, INPUT_DIMENSION), dtype=torch.float32)
            torch.onnx.export(
                net,
                (tensor_x,),
                save_path,
                input_names=["input"],
                output_names=["output"],
                dynamo=True,
                dynamic_axes={
                    "input":  {0: "batch_size"},
                    "output": {0: "batch_size"}
                }
            )
#       print(f"Saved fold {fold} model to {fold_save_path}")
#       print("test6")
#       print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
#       sum = 0.0
#       for key, value in results.items():
#           print(f'Fold {key}: {value} %')
#           sum += value
#       print(f'Average: {sum/len(results.items())} %')

def sigmoid(x):
    # The sigmoid function
    return 1 / (1 + np.exp(-x))

def get_acc(session, x, y):
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Generate predictions for test data
    logits = session.run([output_name], {input_name: x})[0]
    probs = sigmoid(logits) # probabilities from logits
    y_preds = (probs >= 0.5).astype(np.float32)

    return (y_preds == y).mean()

def test_fnn(onnx_path="kfold_models/baseline.onnx"):
    # Create an inference session
    session = ort.InferenceSession(onnx_path)

    # Get data
    x_train, y_train, x_test, y_test = npz_load()
    x_train, y_train = prepare_data(x_train, y_train)
    x_test, y_test = prepare_data(x_test, y_test)

    train_accuracy = get_acc(session, x_train, y_train)
    test_accuracy = get_acc(session, x_test, y_test)

    return test_accuracy, train_accuracy


def test_models(h_layers, h_neurons, lrs, wds, num_iters):
    best = {"test_acc": 0,
            "train_acc": 0, 
            "h_layers": 0, 
            "h_neurons": 0, 
            "lr": 0, 
            "wd": 0, 
            "num_iter": 0}

    for hl in h_layers:
        for hn in h_neurons:
            for lr in lrs:
                for wd in wds:
                    for ni in num_iters:
                        path = get_path(hl, hn, lr, wd, ni)
                        test_acc, train_acc = test_fnn(path)
                        print(f"hl = {hl}, hn = {hn}, lr = {lr}, wd = {wd}, ni = {ni}, train accuracy: {train_acc}, test accuracy: {test_acc}")
            
                        if (test_acc > best["test_acc"]):
                            # new best test accuracy, update all
                            best["test_acc"]  = test_acc
                            best["train_acc"] = train_acc
                            best["h_layers"]  = hl
                            best["h_neurons"] = hn
                            best["lr"] = lr
                            best["wd"] = wd
                            best["num_iter"] = ni
    return best

def generate_models(h_layers, h_neurons, lrs, wds, num_iters):

    for hl in h_layers:
        for hn in h_neurons:
            for lr in lrs:
                for wd in wds:
                    for ni in num_iters:
                        path = get_path(hl, hn, lr, wd, ni)
                        print(f"{path}, ", end="")
                        train_fnn_kfold(num_iter=ni,
                                  hidden_neurons=hn, 
                                  num_hidden_layers=hl, 
                                  learning_rate=lr,
                                  weight_decay=wd,
                                  save_path=path)

if __name__ == "__main__":

    num_iter = [128]
    h_neurons = [1, 2, 4, 8, 16, 32]
    h_layers = [1, 2, 4, 8, 16, 32]
    # h_neurons = [1]
    # h_layers = [1]
    lr = [0.001, 0.0001]
    wd = [0.001, 0.0001]
    
    generate_models(h_layers, h_neurons, lr, wd, num_iter)
    results = test_models(h_layers, h_neurons, lr, wd, num_iter)

    print(results)
