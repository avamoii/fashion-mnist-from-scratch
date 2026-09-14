import numpy as np
from src.utils import load_data
from src.layers import DenseLayer
from src.activations import ReLU, Softmax
from src.network import NeuralNetwork


print("Loading data...")
X_train, Y_train, X_test, Y_test = load_data(
    "data/fashion-mnist_train.csv", 
    "data/fashion-mnist_test.csv"
)


print(f"X_train shape: {X_train.shape}")

layers = [
    DenseLayer(n_in=784, n_out=256),
    ReLU(),
    DenseLayer(n_in=256, n_out=128),
    ReLU(),
    DenseLayer(n_in=128, n_out=64),
    ReLU(),
    DenseLayer(n_in=64, n_out=10),
    Softmax()
]


model = NeuralNetwork(layers)

print("\nPerforming forward pass on X_train...")

X_batch = X_train[:, :32] 
print(f"Input batch shape: {X_batch.shape}") 

predictions = model.forward(X_batch)


print(f"\nPredictions shape: {predictions.shape}") 
print("\nSample predictions for the first image (probabilities):")
print(predictions[:, 0])
print(f"\nSum of probabilities for the first image: {np.sum(predictions[:, 0]):.4f}") 