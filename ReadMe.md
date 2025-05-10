# DaoML: Exploring Neural Networks Inspired by Daoist Philosophy

DaoML is a research project exploring the integration of principles from Daoist philosophy, I-Ching (Yi Jing), and traditional Chinese concepts into modern neural network design and training approaches.

## Research Motivation

This project investigates whether ancient Chinese philosophical systems can offer novel perspectives for machine learning algorithm design. By reimagining neural networks through a Daoist lens, we explore whether networks can achieve greater harmony with data through principles like non-action (Wu Wei), balance (Yin-Yang), and the interconnected transformations of the Five Elements (Wu Xing).

## Core Philosophical Concepts

### 1. Wu Wei (Non-Action)
The principle of "doing without doing" or "effortless action" inspires an optimization approach that selectively updates only the parameters that truly need changing, adapting thresholds dynamically throughout training.

### 2. Lo Shu Magic Square
This ancient 3×3 magic square (洛书) provides a pattern for neural network weight structure, creating a harmonious inductive bias that appears to improve model performance.

### 3. I-Ching (Book of Changes)
The 64 hexagrams from I-Ching offer a framework for tracking and interpreting a network's state during training, with six lines derived from network metrics such as loss trends, accuracy, gradient stability, activation statistics, and weight norms.

### 4. Wu Xing (Five Elements)
The five elements cycle (Water, Wood, Fire, Earth, Metal) inspires learning rate scheduling and optimization strategies, with each element representing different aspects of the training process:
- Water (adaptability): Strengthened by decreasing loss and varied gradients
- Wood (growth): Strengthened by increasing accuracy
- Fire (transformation): Strengthened by high gradient activity
- Earth (stability): Strengthened by balanced metrics
- Metal (precision): Strengthened by high accuracy and low gradients

## Experimental Implementations

This repository contains experimental implementations demonstrating Daoist-inspired neural network concepts:

### 1. Lo Shu Neural Network
An experimental architecture incorporating the Lo Shu magic square as positional significance matrices for weight structure.

### 2. WuWeiOptimizer
An experimental optimizer implementing the principle of Wu Wei (non-action) with selective parameter updates.

### 3. IChingStateTracker
A system that monitors network state during training using six lines derived from network metrics, mapping the training journey through the 64 hexagrams.

### 4. WuXingSystem
A framework based on the Five Elements system to track elemental balance during training.

### 5. CyclicalElementScheduler
An experimental learning rate scheduler based on the Wu Xing five-element cycle.

### 6. DualLayerMechanismOptimizerWrapper
A meta-optimizer combining regular optimization with strategic interventions.

## Preliminary Results

### Lo Shu Neural Network Performance

The experimental Lo Shu neural network shows promising performance compared to a standard neural network on binary classification tasks:

| Metric | Standard NN | Lo Shu NN |
|--------|-------------|-----------|
| Final Test Accuracy | 71.50% | 74.00% |
| Training Time | 1.09s | 0.08s |

The Lo Shu architecture demonstrates both improved accuracy and dramatically faster training (13.6x speedup), suggesting significant computational efficiency benefits.

#### Robustness to Noise:

| Noise Level | Standard NN | Lo Shu NN |
|-------------|-------------|-----------|
| 0.0 | 71.50% | 74.00% |
| 0.1 | 72.00% | 72.50% |
| 0.2 | 69.00% | 72.50% |
| 0.3 | 69.00% | 72.50% |
| 0.4 | 69.00% | 73.00% |
| 0.5 | 69.50% | 73.50% |

The Lo Shu network consistently outperforms the standard network, particularly at higher noise levels, demonstrating greater robustness.

### Complete Framework Results

The integrated framework with Wu Wei optimization, I-Ching tracking, and Wu Xing scheduling achieves promising results:

- Final validation accuracy: 93.33%
- Training accuracy: 99.08%
- Most frequent hexagrams during training:
  - Kun (困, Oppression): 1005 occurrences
  - Sun (巽, The Gentle): 684 occurrences
  - Da Xu (大畜, Great Taming): 608 occurrences

## Example Files

The repository includes example implementations demonstrating these concepts:

1. `examples/lo_shu_nn_00/` - Lo Shu neural network vs. standard neural network comparison
2. `examples/wuwei_optimizer/` - Complete integration with Wu Xing and I-Ching tracking
3. Visualizations of training dynamics and elemental balances

## Implementing the Concepts

Below is a simplified example of how the Lo Shu neural network is implemented in our experiments:

```python
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
```

And a simplified example of the Wu Wei optimization concept:

```python
# Using the optimizer in practice
model = StandardNN()  # or LoShuNN()
optimizer = torch.optim.Adam(model.parameters())

# For each training step
for inputs, targets in dataloader:
    # Forward and calculate loss
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Track metrics for I-Ching state and Wu Xing balance
    metrics = {
        'loss': loss.item(),
        'accuracy': calculate_accuracy(outputs, targets),
        'gradient_norm': calculate_gradient_norm(model),
        'loss_decreasing': is_loss_decreasing()
    }
    
    # Update element strengths based on metrics
    wu_xing_system.update_batch(model, metrics)
    
    # Update hexagram state
    hexagram_tracker.update_state(metrics)
    
    # Optimize with selective updates
    optimizer.step()
```

---

```
Nothing that rises.
Will not also fall.

Nothing that falls will not also rise.

What rises is Fire, what descends is Water.
That which seeks to rise but cannot is Wood.
That which seeks to fall but cannot is Metal.
That which neither rises nor falls but stabilizes both is Earth.
```

---

## Citation

If you use these concepts in your research, please cite:

```
@software{daoml2025,
  author = {Maximilian Winter},
  title = {DaoML: Exploring Neural Networks Inspired by Daoist Philosophy},
  url = {https://github.com/Maximilian-Winter/daoml},
  year = {2025},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.