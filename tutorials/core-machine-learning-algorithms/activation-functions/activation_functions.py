import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    exp_x = np.exp(x)
    exp_neg_x = np.exp(-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example usage
if __name__ == "__main__":
    # Generate synthetic input data
    np.random.seed(0)

    # For Sigmoid, Tanh, ReLU, Leaky ReLU: single array
    x = np.random.randn(5)  # 5 scalar inputs

    # For Softmax: simulate scores for 3 classes
    x_softmax = np.random.randn(5, 3)  # 5 samples, 3 classes

    # Calculate activations
    sigmoid_output = sigmoid(x)
    tanh_output = tanh(x)
    relu_output = relu(x)
    leaky_relu_output = leaky_relu(x, alpha=0.01)
    softmax_output = softmax(x_softmax)

    # Print results
    print("Input (x):", x)
    print("Sigmoid:", sigmoid_output)
    print("Tanh:", tanh_output)
    print("ReLU:", relu_output)
    print("Leaky ReLU (alpha=0.01):", leaky_relu_output)
    print("Softmax Input (5 samples, 3 classes):", x_softmax)
    print("Softmax Output:", softmax_output)
