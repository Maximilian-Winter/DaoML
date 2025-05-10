import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define Lo Shu magic square
lo_shu = torch.tensor([
    [4, 9, 2],
    [3, 5, 7],
    [8, 1, 6]
]).float()

# Normalize to range [0.5, 1.0] to avoid zeroing out connections
lo_shu_normalized = 0.5 + 0.5 * (lo_shu - lo_shu.min()) / (lo_shu.max() - lo_shu.min())


# Create a positional significance matrix
def create_positional_matrix(input_size, output_size):
    pos_matrix = torch.ones(output_size, input_size)

    # For a 9x9 layer, apply Lo Shu significance
    if input_size == 9 and output_size == 9:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        # Map (i,j,k,l) to position in the weight matrix
                        input_idx = i * 3 + j
                        output_idx = k * 3 + l

                        # Calculate positional significance
                        input_sig = lo_shu_normalized[i, j]
                        output_sig = lo_shu_normalized[k, l]

                        # Multiply significance factors
                        pos_matrix[output_idx, input_idx] = input_sig * output_sig

    return pos_matrix


# Standard neural network
class StandardNN(nn.Module):
    def __init__(self, input_size=9, hidden_size=9, output_size=1):
        super(StandardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Lo Shu neural network
class LoShuNN(nn.Module):
    def __init__(self, input_size=9, hidden_size=9, output_size=1):
        super(LoShuNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # Create positional significance matrices
        self.pos_matrix1 = create_positional_matrix(input_size, hidden_size)
        self.pos_matrix2 = create_positional_matrix(hidden_size, output_size)

    def forward(self, x):
        # Apply positional significance during forward pass
        w1 = self.fc1.weight * self.pos_matrix1.to(self.fc1.weight.device)
        x = self.relu(nn.functional.linear(x, w1, self.fc1.bias))

        w2 = self.fc2.weight * self.pos_matrix2.to(self.fc2.weight.device)
        x = nn.functional.linear(x, w2, self.fc2.bias)
        return x


# Generate synthetic data
def generate_data(n_samples=1000):
    # Create classification problem with 9 features (to match Lo Shu dimensions)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=9,
        n_informative=6,
        n_redundant=3,
        random_state=42
    )

    # Make binary classification (0 or 1)
    y = y.reshape(-1, 1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test


# Training function
def train_model(model, X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # For storing metrics
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)

        # Backward pass and optimize
        train_loss.backward()
        optimizer.step()

        # Calculate training accuracy
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        train_accuracy = (predicted == y_train).float().mean()

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)

            # Calculate test accuracy
            test_predicted = (torch.sigmoid(test_outputs) >= 0.5).float()
            test_accuracy = (test_predicted == y_test).float().mean()

        # Store metrics
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        train_accuracies.append(train_accuracy.item())
        test_accuracies.append(test_accuracy.item())

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, '
                f'Train Acc: {train_accuracy.item():.4f}, Test Acc: {test_accuracy.item():.4f}')

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }


# Visualization function
def plot_results(standard_metrics, loshu_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot training loss
    axes[0, 0].plot(standard_metrics['train_losses'], label='Standard NN')
    axes[0, 0].plot(loshu_metrics['train_losses'], label='Lo Shu NN')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Plot test loss
    axes[0, 1].plot(standard_metrics['test_losses'], label='Standard NN')
    axes[0, 1].plot(loshu_metrics['test_losses'], label='Lo Shu NN')
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()

    # Plot training accuracy
    axes[1, 0].plot(standard_metrics['train_accuracies'], label='Standard NN')
    axes[1, 0].plot(loshu_metrics['train_accuracies'], label='Lo Shu NN')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()

    # Plot test accuracy
    axes[1, 1].plot(standard_metrics['test_accuracies'], label='Standard NN')
    axes[1, 1].plot(loshu_metrics['test_accuracies'], label='Lo Shu NN')
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


# Visualize weight patterns
def visualize_weights(standard_model, loshu_model):
    # Get weights from the first layer
    standard_weights = standard_model.fc1.weight.detach().numpy()
    loshu_weights = loshu_model.fc1.weight.detach().numpy()

    # Create position significance matrix
    pos_matrix = create_positional_matrix(9, 9).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot standard weights
    im1 = axes[0].imshow(standard_weights, cmap='viridis')
    axes[0].set_title('Standard NN Weights')
    axes[0].set_xlabel('Input Neuron')
    axes[0].set_ylabel('Hidden Neuron')
    plt.colorbar(im1, ax=axes[0])

    # Plot Lo Shu weights
    im2 = axes[1].imshow(loshu_weights, cmap='viridis')
    axes[1].set_title('Lo Shu NN Weights')
    axes[1].set_xlabel('Input Neuron')
    axes[1].set_ylabel('Hidden Neuron')
    plt.colorbar(im2, ax=axes[1])

    # Plot positional significance matrix
    im3 = axes[2].imshow(pos_matrix, cmap='viridis')
    axes[2].set_title('Lo Shu Positional Significance')
    axes[2].set_xlabel('Input Neuron')
    axes[2].set_ylabel('Hidden Neuron')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()


# Function to evaluate model robustness to noise
def evaluate_robustness(model, X_test, y_test, noise_levels):
    accuracies = []

    for noise_level in noise_levels:
        noisy_X = X_test + torch.randn_like(X_test) * noise_level

        model.eval()
        with torch.no_grad():
            outputs = model(noisy_X)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            accuracy = (predicted == y_test).float().mean().item()

        accuracies.append(accuracy)

    return accuracies


# Main function
def main():
    # Generate data
    print("Generating synthetic data...")
    X_train, y_train, X_test, y_test = generate_data()

    # Initialize models
    print("Initializing models...")
    standard_model = StandardNN()
    loshu_model = LoShuNN()

    # Train standard model
    print("\nTraining Standard Neural Network...")
    start_time = time.time()
    standard_metrics = train_model(standard_model, X_train, y_train, X_test, y_test, epochs=100)
    standard_time = time.time() - start_time
    print(f"Standard model training completed in {standard_time:.2f} seconds")

    # Train Lo Shu model
    print("\nTraining Lo Shu Neural Network...")
    start_time = time.time()
    loshu_metrics = train_model(loshu_model, X_train, y_train, X_test, y_test, epochs=100)
    loshu_time = time.time() - start_time
    print(f"Lo Shu model training completed in {loshu_time:.2f} seconds")

    # Plot training results
    print("\nPlotting results...")
    plot_results(standard_metrics, loshu_metrics)

    # Visualize weights
    visualize_weights(standard_model, loshu_model)

    # Evaluate robustness to noise
    print("\nEvaluating robustness to noise...")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    standard_robustness = evaluate_robustness(standard_model, X_test, y_test, noise_levels)
    loshu_robustness = evaluate_robustness(loshu_model, X_test, y_test, noise_levels)

    # Plot robustness comparison
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, standard_robustness, 'o-', label='Standard NN')
    plt.plot(noise_levels, loshu_robustness, 's-', label='Lo Shu NN')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy')
    plt.title('Model Robustness to Noise')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print final performance comparison
    print("\nFinal Performance Comparison:")
    print(f"Standard NN - Final Test Accuracy: {standard_metrics['test_accuracies'][-1]:.4f}")
    print(f"Lo Shu NN - Final Test Accuracy: {loshu_metrics['test_accuracies'][-1]:.4f}")
    print(f"\nTraining Time - Standard NN: {standard_time:.2f}s, Lo Shu NN: {loshu_time:.2f}s")

    # Print robustness comparison
    print("\nRobustness to Noise:")
    for i, noise in enumerate(noise_levels):
        print(f"Noise Level {noise:.1f} - Standard: {standard_robustness[i]:.4f}, Lo Shu: {loshu_robustness[i]:.4f}")


if __name__ == "__main__":
    main()