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
    # Batch version (recommended)
    x_tensor = torch.from_numpy(x_train).float()   # shape (N, INPUT_DIMENSION)
    y_tensor = torch.from_numpy(y_train).float()   # shape (N,1)

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
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}} #Allowing for variable size accesing
                  )
def sigmoid(x):
    # The sigmoid function
    return 1 / (1 + np.exp(-x))

def test_fnn(onnx_path="models/baseline.onnx"):
    # Create an inference session
    session = ort.InferenceSession(onnx_path)

    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Get data
    _, _, x_test, y_test = npz_load()
    x_test, y_test = prepare_data(x_test, y_test)

    # Generate predictions for test data
    logits = session.run([output_name], {input_name: x_test})[0]
    probs = sigmoid(logits) # probabilities from logits
    y_preds = (probs >= 0.5).astype(np.float32)

    accuracy = (y_preds == y_test).mean()
    print(f"Test accuracy: {accuracy*100:.2f}%")

    return accuracy

def get_path(hlayers, hneurons, lr, wd, iter):
    return f"models/baseline_{hlayers}_hlayers_{hneurons}_hneurons_{lr}_{wd}_{iter}.onnx"

if __name__ == "__main__":

    # h_neur = 4
    # n_h_layers = 2
    lr = 0.001
    wd = 1e-4
    num_iter = 2

    h_nuers = [2]
    h_layers_opts = [2]
    for h_neur in h_nuers:
        for n_h_layers in h_layers_opts:
            path = get_path(n_h_layers, h_neur, lr, wd, num_iter)
            train_fnn(num_iter=num_iter,
                      hidden_neurons=h_neur, 
                      num_hidden_layers=n_h_layers, 
                      learning_rate=lr,
                      weight_decay=wd,
                      save_path=path)
    
    # test_fnn(get_path(n_h_layers, h_neur, lr, wd, num_iter))