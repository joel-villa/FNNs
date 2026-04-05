import torch
import time
import numpy as np
from load_data import npz_load
import onnxruntime as ort

# 5,000 features -> first layer has 5000 input neurons
INPUT_DIMENSION = 5000

# Binary classification -> one neuron in output layer
OUTPUT_NEURONS = 1

def make_fnn(h_neurons, num_hidden_layers):
    layers = []

    # First layer
    layers.append(torch.nn.Linear(INPUT_DIMENSION, h_neurons))
    layers.append(torch.nn.ReLU())

    # Hidden layers
    for _ in range(num_hidden_layers - 1):
        layers.append(torch.nn.Linear(h_neurons, h_neurons))
        layers.append(torch.nn.ReLU())

    # Output layer
    layers.append(torch.nn.Linear(h_neurons, OUTPUT_NEURONS))

    return torch.nn.Sequential(*layers)

def prepare_data(x, y):
    x = x.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1) # y is now explicitly an nx1 array

    return x, y

def train_fnn(num_iter, num_hidden_layers, hidden_neurons, learning_rate, weight_decay, save_path=None):

    net = make_fnn(h_neurons=hidden_neurons, num_hidden_layers=num_hidden_layers)

    # Get data
    x_train, y_train, _, _ = npz_load()

    x_train, y_train = prepare_data(x_train, y_train)
    

    optimizer = torch.optim.Adam(net.parameters(),
                                lr=learning_rate, 
                                weight_decay=weight_decay)
    
    L = torch.nn.BCEWithLogitsLoss()

    start = time.perf_counter() #Timing training
    # Batch version 
    x_tensor = torch.from_numpy(x_train).float()  
    y_tensor = torch.from_numpy(y_train).float()  

    batch_size = 32
    for epoch in range(num_iter):
        for i in range(0, len(x_tensor), batch_size):
            xb = x_tensor[i:i+batch_size]
            yb = y_tensor[i:i+batch_size]

            output = net.forward(xb)
            loss = L(output, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    end = time.perf_counter()

    print(f"Train time: {end - start:.6f} seconds")


    # Saving in ONNX file
    net.eval()
    tensor_x = torch.rand((1, INPUT_DIMENSION), dtype=torch.float32)
    torch.onnx.export(net,                 # model to export
                  (tensor_x,),             # inputs of the model,
                  save_path,               # filename of the ONNX model
                  input_names=["input"],   # Rename inputs for the ONNX model
                  output_names=["output"], # Rename output
                  dynamo=True,             # True or False to select the exporter to use
                  dynamic_shapes={"input": {0: torch.export.Dim("batch")}} #Allowing for variable size accesing
                  # "output": {0: "batch_size"}
                  )
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

def test_fnn(onnx_path="models/baseline.onnx"):
    # Create an inference session
    session = ort.InferenceSession(onnx_path)

    # Get data
    x_train, y_train, x_test, y_test = npz_load()
    x_train, y_train = prepare_data(x_train, y_train)
    x_test, y_test = prepare_data(x_test, y_test)

    train_accuracy = get_acc(session, x_train, y_train)
    test_accuracy = get_acc(session, x_test, y_test)

    return test_accuracy, train_accuracy

def get_path(hlayers, hneurons, lr, wd, iter):
    return f"models/baseline_{hlayers}_hlayers_{hneurons}_hneurons_{lr}_{wd}_{iter}.onnx"

def test_models(h_layers, h_neurons, lr, wd, num_iter):
    best = {"test_acc": 0,
            "train_acc": 0, 
            "h_layers": 0, 
            "h_nuerons": 0, 
            "lr": lr, 
            "wd": wd, 
            "num_iter": num_iter}

    for hl in h_layers:
        for hn in h_neurons:
            path = get_path(hl, hn, lr, wd, num_iter)
            test_acc, train_acc = test_fnn(path)
            if (train_acc > best["train_acc"]):
                best["train_acc"] = train_acc
                best["test_acc"]  = test_acc
                best["h_layers"]  = hl
                best["h_nuerons"] = hn
    return best

if __name__ == "__main__":
    lr = 0.001
    wd = 1e-4
    num_iter = 32

    h_neurons = [2, 4, 8, 16]
    h_layers = [2, 4, 8, 16, 32]

    # h_neurons = [2]
    # h_layers = [2]
    for hn in h_neurons:
        for hl in h_layers:
            path = get_path(hl, hn, lr, wd, num_iter)
            print(f"{path}, ", end="")
            train_fnn(num_iter=num_iter,
                      hidden_neurons=hn, 
                      num_hidden_layers=hl, 
                      learning_rate=lr,
                      weight_decay=wd,
                      save_path=path)
    
    #results = test_models(h_layers, h_neurons, lr, wd, num_iter)
    #print(results)

    # test_fnn(get_path(2, 2, lr, wd, num_iter))
