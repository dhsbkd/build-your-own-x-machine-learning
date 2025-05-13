import numpy as np

def mean_absolute_error(y_true, y_pred):

    # Calculate MAE: mean of absolute differences
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(0)
    X = np.random.randn(100, 1)  # 100 samples, 1 feature
    y_true = 3 * X.flatten() + 2 + np.random.randn(100) * 0.5  # y = 3x + 2 + noise

    # Simulate predictions
    # Assume a slightly off model: y_pred = 2.8x + 2.2
    y_pred = 2.8 * X.flatten() + 2.2

    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)

    # Print results
    print("Actual values (first 5):", y_true[:5])
    print("Predicted values (first 5):", y_pred[:5])
    print("Mean Absolute Error:", mae)
