import torch
import time
import numpy as np
from load_data import npz_load
import onnxruntime as ort

from baseline import get_path, make_fnn, prepare_data

# 5,000 features -> first layer has 5000 input neurons
INPUT_DIMENSION = 5000

# Binary classification -> one neuron in output layer
OUTPUT_NEURONS = 1

# currenly just copied over from baseline
# TODO: implement kfold cross-validation

def train_fnn_kfold(num_iter, num_hidden_layers, hidden_neurons, learning_rate, weight_decay, save_path=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = make_fnn(h_neurons=hidden_neurons, num_hidden_layers=num_hidden_layers).to(device)

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
    # x_tensor = torch.from_numpy(x_train).float().to(device)   # shape (N, INPUT_DIMENSION)
    # y_tensor = torch.from_numpy(y_train).float().to(device)   # shape (N,1)
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
    # tensor_x = torch.rand((1, INPUT_DIMENSION), dtype=torch.float32, device=device)
    tensor_x = torch.rand((1, INPUT_DIMENSION), dtype=torch.float32)
    torch.onnx.export(net,                 # model to export
                  (tensor_x,),             # inputs of the model,
                  save_path,               # filename of the ONNX model
                  input_names=["input"],   # Rename inputs for the ONNX model
                  output_names=["output"], # Rename output
                  dynamo=True,             # True or False to select the exporter to use
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}} #Allowing for variable size accesing
                  )

if __name__ == "__main__":
    lr = 0.001
    wd = 1e-4
    num_iter = 32

    h_nuers = [2, 4, 8, 16]
    h_layers_opts = [2, 4, 8, 16, 32]
    for h_neur in h_nuers:
        for n_h_layers in h_layers_opts:
            path = get_path(n_h_layers, h_neur, lr, wd, num_iter)
            train_fnn(num_iter=num_iter,
                      hidden_neurons=h_neur, 
                      num_hidden_layers=n_h_layers, 
                      learning_rate=lr,
                      weight_decay=wd,
                      save_path=path)