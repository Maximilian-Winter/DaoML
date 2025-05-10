import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import DaoML components
from i_ching import Hexagrams, IChingStateTracker
from wu_xing import WuXingSystem, Element


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Define the WuWeiOptimizer class
class WuWeiOptimizer(torch.optim.Optimizer):
    """
    Enhanced Wu Wei Optimizer with dynamic threshold adaptation and strategic non-action.

    This optimizer embodies the principles of Wu Wei (non-action) by:
    1. Only updating parameters that truly need changing
    2. Adapting thresholds based on training progress
    3. Preserving natural parameter flows while removing obstacles
    4. Employing a three-tier update strategy (strong, moderate, none)
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999),
                 threshold=0.01, adapt_rate=0.01, wu_wei_ratio=0.7):
        """
        Initialize the enhanced Wu Wei Optimizer.

        Args:
            params: Model parameters (list or iterator)
            lr: Learning rate
            betas: Coefficients for computing running averages
            threshold: Initial gradient threshold for parameter updates
            adapt_rate: Rate at which thresholds adapt (0-1)
            wu_wei_ratio: Proportion of parameters to leave unchanged (0-1)
        """
        # Ensure params is a list to avoid empty parameter issues
        if not isinstance(params, (list, tuple)):
            params = list(params)

        # Validate params is not empty
        if len(params) == 0:
            raise ValueError("Optimizer got an empty parameter list")

        defaults = dict(lr=lr, betas=betas, threshold=threshold,
                        adapt_rate=adapt_rate, wu_wei_ratio=wu_wei_ratio)

        super(WuWeiOptimizer, self).__init__(params, defaults)

        # Inner optimizer for standard updates (Adam)
        # Use a new instance with the SAME parameter list
        self.inner_optimizer = torch.optim.Adam(self.param_groups, lr=lr, betas=betas)

        # Counter for tracking step number
        self.step_count = 0

        # Tracking historical gradient significance for adaptive thresholds
        self.grad_history = {}
        self.history_window = 50  # Window size for historical tracking

        # Parameter-specific thresholds (initialized on first step)
        self.param_thresholds = {}

        # Performance metrics
        self.update_ratios = []  # Track percentage of parameters updated

    def zero_grad(self, set_to_none=False):
        """Clear gradients of all parameters"""
        # Call parent method to clear our gradients
        super().zero_grad(set_to_none=set_to_none)

        # Also clear gradients in inner optimizer
        self.inner_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """
        Perform a single optimization step with Wu Wei principles.

        Args:
            closure: Function that evaluates the model and returns loss

        Returns:
            Loss value
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Increment step count
        self.step_count += 1

        # Apply inner optimizer to get proposed updates
        self.inner_optimizer.step()

        # Apply Wu Wei principle to selectively keep or revert updates
        update_count = 0
        total_params = 0

        for group in self.param_groups:
            threshold = group['threshold']
            adapt_rate = group['adapt_rate']
            wu_wei_ratio = group['wu_wei_ratio']

            for p in group['params']:
                if p.grad is None:
                    continue

                total_params += p.numel()

                # Get parameter state
                state = self.state[p]

                # Store previous parameters if not already stored
                if 'prev_params' not in state:
                    state['prev_params'] = p.data.clone()

                # Get or initialize parameter-specific threshold
                param_id = id(p)
                if param_id not in self.param_thresholds:
                    self.param_thresholds[param_id] = threshold

                # Calculate gradient significance
                if len(p.shape) > 1:  # For matrices, calculate per-output significance
                    grad_significance = p.grad.abs().mean(dim=1)

                    # Get proposed updates
                    proposed_update = p.data - state['prev_params']
                    update_significance = proposed_update.abs().mean(dim=1)

                    # Determine which outputs to update based on thresholds
                    current_threshold = self.param_thresholds[param_id]
                    update_mask = (grad_significance > current_threshold).float().unsqueeze(1)

                    # Apply selective update with three tiers
                    high_mask = (grad_significance > 2 * current_threshold).float().unsqueeze(1)
                    med_mask = ((grad_significance > current_threshold) &
                                (grad_significance <= 2 * current_threshold)).float().unsqueeze(1)

                    # Reset to previous values
                    p.data.copy_(state['prev_params'])

                    # Apply tiered updates: 100% for high, 50% for medium, 0% for low
                    p.data += high_mask * proposed_update
                    p.data += med_mask * proposed_update * 0.5

                    # Count updated parameters
                    update_count += (update_mask.sum() * p.shape[1]).item()

                else:  # For vectors, use element-wise significance
                    grad_significance = p.grad.abs()

                    # Get proposed updates
                    proposed_update = p.data - state['prev_params']
                    update_significance = proposed_update.abs()

                    # Determine which elements to update based on thresholds
                    current_threshold = self.param_thresholds[param_id]
                    update_mask = (grad_significance > current_threshold).float()

                    # Apply selective update with three tiers
                    high_mask = (grad_significance > 2 * current_threshold).float()
                    med_mask = ((grad_significance > current_threshold) &
                                (grad_significance <= 2 * current_threshold)).float()

                    # Reset to previous values
                    p.data.copy_(state['prev_params'])

                    # Apply tiered updates: 100% for high, 50% for medium, 0% for low
                    p.data += high_mask * proposed_update
                    p.data += med_mask * proposed_update * 0.5

                    # Count updated parameters
                    update_count += update_mask.sum().item()

                # Update parameter-specific thresholds based on gradient history
                if param_id not in self.grad_history:
                    self.grad_history[param_id] = []

                self.grad_history[param_id].append(p.grad.abs().mean().item())
                if len(self.grad_history[param_id]) > self.history_window:
                    self.grad_history[param_id].pop(0)

                # Adapt threshold using gradient history
                if len(self.grad_history[param_id]) > 10:
                    recent_grads = self.grad_history[param_id][-10:]
                    median_grad = sorted(recent_grads)[len(recent_grads) // 2]

                    # Adjust threshold to maintain approximately wu_wei_ratio of non-updates
                    if update_count / max(1, total_params) > (1 - wu_wei_ratio):
                        # Too many updates, increase threshold
                        self.param_thresholds[param_id] += adapt_rate * median_grad
                    else:
                        # Too few updates, decrease threshold
                        self.param_thresholds[param_id] = max(0.001,
                                                              self.param_thresholds[
                                                                  param_id] - adapt_rate * median_grad)

                # Store current parameters for next iteration
                state['prev_params'] = p.data.clone()

        # Track update ratio for monitoring
        if total_params > 0:
            self.update_ratios.append(update_count / total_params)

            # Keep only recent history
            if len(self.update_ratios) > 100:
                self.update_ratios = self.update_ratios[-100:]

        return loss

    def step_with_trackers(self, trackers):
        """
        Enhanced step method that incorporates tracker information for Wu Wei optimization.

        Args:
            trackers: Dictionary of metric trackers

        Returns:
            Loss value
        """
        # Adjust wu_wei_ratio based on hexagram state
        if 'hexagram_state' in trackers:
            hex_tracker = trackers['hexagram_state']
            if hasattr(hex_tracker, 'get_current_hexagram_index'):
                hex_idx = hex_tracker.get_current_hexagram_index()

                # Adjust wu_wei_ratio based on hexagram
                for group in self.param_groups:
                    if hex_idx == 1:  # The Creative (Qian) - pure yang
                        # More action, less wu-wei
                        group['wu_wei_ratio'] = 0.3
                    elif hex_idx == 2:  # The Receptive (Kun) - pure yin
                        # More wu-wei, less action
                        group['wu_wei_ratio'] = 0.8
                    elif hex_idx == 11:  # Peace (Tai) - balanced
                        # Balanced approach
                        group['wu_wei_ratio'] = 0.5
                    elif hex_idx == 12:  # Standstill (Pi) - stagnation
                        # Much less wu-wei to escape stagnation
                        group['wu_wei_ratio'] = 0.2

        # Adjust thresholds based on Wu Xing balance
        if 'wu_xing_balance' in trackers:
            wu_xing_tracker = trackers['wu_xing_balance']
            if hasattr(wu_xing_tracker, 'get_element_strengths'):
                element_strengths = wu_xing_tracker.get_element_strengths()

                # Get dominant element
                dominant_element = max(element_strengths.items(), key=lambda x: x[1])[0]

                # Adjust threshold and adapt_rate based on dominant element
                for group in self.param_groups:
                    if dominant_element == Element.WATER:  # Flowing, adaptable
                        group['adapt_rate'] = 0.02  # More adaptive
                    elif dominant_element == Element.WOOD:  # Growing, expanding
                        group['threshold'] *= 0.9  # Lower threshold
                    elif dominant_element == Element.FIRE:  # Transformative, active
                        group['wu_wei_ratio'] = 0.4  # More action
                    elif dominant_element == Element.EARTH:  # Stabilizing, grounding
                        group['wu_wei_ratio'] = 0.6  # More stability
                    elif dominant_element == Element.METAL:  # Refining, precise
                        group['threshold'] *= 1.1  # Higher threshold for precision

        # Perform standard step
        return self.step()


# Define the EnhancedMechanismGrabber class (basic version for example)
class EnhancedMechanismGrabber:
    """
    Enhanced implementation of the mechanism-grabbing principle with improved efficiency.
    Identifies critical points with minimal computational overhead and maximizes intervention impact.
    """

    def __init__(self):
        """Initialize the enhanced mechanism grabber"""
        self.hexagrams = Hexagrams()
        self.wu_xing = WuXingSystem()

        # Simplified implementation for example
        self.intervention_types = {
            'activate': self._activate_intervention,
            'stabilize': self._stabilize_intervention,
            'balance': self._balance_intervention
        }

    def find_mechanism_points(self, model, trackers, metrics=None):
        """Find mechanism points for intervention"""
        # This is a simplified version
        mechanism_points = []

        # Simply return some placeholder values for the example
        for i, (name, module) in enumerate(model.named_modules()):
            if isinstance(module, nn.Linear):
                # Add a potential mechanism point
                strength = 1.0 + 0.1 * i  # Simple strength calculation

                intervention = {
                    'type': 'balance',
                    'layer': name,
                    'description': 'Balance weight distribution'
                }

                mechanism_points.append((name, strength, intervention))

        # Sort by strength
        mechanism_points.sort(key=lambda x: x[1], reverse=True)

        return mechanism_points[:3]  # Return top 3 points

    def apply_intervention(self, model, intervention):
        """Apply an intervention"""
        # Get intervention type and layer
        int_type = intervention['type']
        layer_name = intervention['layer']

        # Get the layer
        layer = None
        for name, module in model.named_modules():
            if name == layer_name:
                layer = module
                break

        if layer is None:
            return False

        # Apply the intervention
        if int_type in self.intervention_types:
            return self.intervention_types[int_type](layer, intervention)

        return False

    def _activate_intervention(self, layer, intervention):
        """Simple activation intervention"""
        if hasattr(layer, 'weight') and layer.weight is not None:
            with torch.no_grad():
                # Find near-zero weights
                mask = (layer.weight.abs() < 0.01)

                # Add small random values to dead weights
                if mask.sum() > 0:
                    layer.weight.data[mask] += torch.randn_like(layer.weight.data[mask]) * 0.1
                    return True
        return False

    def _stabilize_intervention(self, layer, intervention):
        """Simple stabilization intervention"""
        if hasattr(layer, 'weight') and layer.weight is not None:
            with torch.no_grad():
                # Calculate weight statistics
                weight_mean = layer.weight.mean()
                weight_std = layer.weight.std()

                # Find outlier weights
                mask = (layer.weight - weight_mean).abs() > 2 * weight_std

                # Move outliers toward the mean
                if mask.sum() > 0:
                    layer.weight.data[mask] = layer.weight.data[mask] * 0.7 + weight_mean * 0.3
                    return True
        return False

    def _balance_intervention(self, layer, intervention):
        """Simple balance intervention"""
        if hasattr(layer, 'weight') and layer.weight is not None:
            with torch.no_grad():
                # Balance positive and negative weights
                pos_sum = layer.weight[layer.weight > 0].sum()
                neg_sum = layer.weight[layer.weight < 0].sum()

                if abs(pos_sum + neg_sum) > 0.1:
                    # Scale to balance
                    if neg_sum.abs() > 0:
                        layer.weight.data[layer.weight < 0] *= pos_sum.abs() / neg_sum.abs()
                        return True
        return False


# Define the scheduler classes
class CyclicalElementScheduler:
    """
    Learning rate scheduler based on the Wu Xing five-element cycle.
    Follows the natural generation and conquest cycles of the five elements.
    """

    def __init__(self, optimizer, cycle_length=100, base_decay=0.95, min_lr=1e-6, max_lr=1e-2):
        """
        Initialize CyclicalElementScheduler with improved strategy.

        Args:
            optimizer: Optimizer to schedule (already initialized)
            cycle_length: Length of a complete five-element cycle
            base_decay: Base learning rate decay factor
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
        """
        # Validate optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")

        self.optimizer = optimizer
        self.cycle_length = cycle_length
        self.base_decay = base_decay
        self.min_lr = min_lr
        self.max_lr = max_lr

        # Initialize state tracking
        self.step_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.wu_xing = WuXingSystem()
        self.current_element = Element.WATER  # Start with Water

        # Element-specific learning rate factors
        self.element_factors = {
            Element.WATER: 1.0,  # Balanced, flowing
            Element.WOOD: 1.2,  # Growth, expansion
            Element.FIRE: 1.5,  # Transformation, activity
            Element.EARTH: 0.8,  # Stability, consolidation
            Element.METAL: 0.6  # Refinement, precision
        }

    def step(self, metrics=None):
        """Basic step with cycle"""
        self.step_count += 1

        # Calculate element phase position
        phase_length = self.cycle_length // 5
        cycle_position = self.step_count % self.cycle_length
        phase_index = cycle_position // phase_length

        # Map phase index to element
        elements = [Element.WATER, Element.WOOD, Element.FIRE, Element.EARTH, Element.METAL]
        current_element = elements[phase_index]

        # Get element factor
        element_factor = self.element_factors[current_element]

        # Apply base decay
        decay_factor = self.base_decay ** (self.step_count / 100)

        # Combined factor
        combined_factor = element_factor * decay_factor

        # Update learning rates with bounds
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = max(self.min_lr, min(self.max_lr, self.base_lrs[i] * combined_factor))

        # Update current element
        self.current_element = current_element

        return [group['lr'] for group in self.optimizer.param_groups]

    def update_with_metrics(self, loss, accuracy, trackers):
        """Update learning rate based on metrics and Wu Xing balance"""
        # Apply standard step
        current_lrs = self.step()

        # If Wu Xing tracker is available, adjust based on element balance
        if 'wu_xing_balance' in trackers:
            wu_xing_tracker = trackers['wu_xing_balance']
            if hasattr(wu_xing_tracker, 'get_element_strengths'):
                # Get current element strengths
                element_strengths = wu_xing_tracker.get_element_strengths()

                # Find dominant element
                dominant_element = max(element_strengths.items(), key=lambda x: x[1])[0]

                # Apply element-specific multiplier
                multiplier = 1.0
                if dominant_element == Element.WATER:
                    multiplier = 1.0  # Balanced
                elif dominant_element == Element.WOOD:
                    multiplier = 1.2  # Growth
                elif dominant_element == Element.FIRE:
                    multiplier = 1.5  # Transformation
                elif dominant_element == Element.EARTH:
                    multiplier = 0.8  # Stability
                elif dominant_element == Element.METAL:
                    multiplier = 0.6  # Refinement

                # Apply to learning rates
                for group in self.optimizer.param_groups:
                    group['lr'] = max(self.min_lr, min(self.max_lr, group['lr'] * multiplier))

        return [group['lr'] for group in self.optimizer.param_groups]


# Define the DualLayerMechanismOptimizerWrapper class
class DualLayerMechanismOptimizerWrapper:
    """
    A meta-optimizer that combines regular optimization with mechanism-based interventions.

    This approach divides the optimization process into two complementary layers:
    1. Regular, continuous updates using an inner optimizer (Adam, SGD, etc.)
    2. Strategic, targeted interventions at mechanism points using EnhancedMechanismGrabber
    """

    def __init__(self, model, base_optimizer, mechanism_grabber=None,
                 intervention_frequency=10, mechanism_threshold=1.5,
                 scheduler=None):
        """
        Initialize the dual-layer optimizer.

        Args:
            model: The neural network model
            base_optimizer: The inner optimizer for continuous updates
            mechanism_grabber: Mechanism grabber for strategic interventions (optional)
            intervention_frequency: How often to check for mechanism points
            mechanism_threshold: Threshold for applying interventions
            scheduler: Learning rate scheduler (optional)
        """
        # Validate inputs
        if not hasattr(model, 'parameters'):
            raise TypeError("Model must have parameters() method")

        if not isinstance(base_optimizer, torch.optim.Optimizer):
            raise TypeError("base_optimizer must be a torch.optim.Optimizer instance")

        self.model = model
        self.base_optimizer = base_optimizer
        self.mechanism_grabber = mechanism_grabber or EnhancedMechanismGrabber()
        self.intervention_frequency = intervention_frequency
        self.mechanism_threshold = mechanism_threshold
        self.scheduler = scheduler

        # Initialize state tracking
        self.step_count = 0
        self.trackers = {}
        self.intervention_history = []
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'hexagram_states': [],
            'element_states': [],
            'mechanism_points': [],
            'interventions': []
        }

    def register_trackers(self, trackers):
        """Register metric trackers for the optimizer"""
        self.trackers = trackers
        return self

    def zero_grad(self, set_to_none=False):
        """
        Clear gradients of all parameters

        Args:
            set_to_none: boolean that determines the behavior (set to zero or None)
        """
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, loss=None, accuracy=None, closure=None):
        """
        Perform a single optimization step with dual-layer approach.

        Args:
            loss: Current loss value (optional)
            accuracy: Current accuracy value (optional)
            closure: Function that evaluates the model and returns loss (optional)

        Returns:
            Loss value
        """
        # Increment step count
        self.step_count += 1

        # Add metrics to history if provided
        if loss is not None:
            self.metrics['train_loss'].append(loss)
            if len(self.metrics['train_loss']) > 2:
                loss_decreasing = self.metrics['train_loss'][-1] < self.metrics['train_loss'][-2]
            else:
                loss_decreasing = True
        else:
            loss_decreasing = True

        if accuracy is not None:
            self.metrics['train_accuracy'].append(accuracy)

        # Execute inner optimizer step
        inner_loss = None
        if closure is not None:
            inner_loss = closure()

        # Apply inner optimizer step
        if hasattr(self.base_optimizer, 'step_with_trackers') and self.trackers:
            self.base_optimizer.step_with_trackers(self.trackers)
        else:
            self.base_optimizer.step()

        # Apply scheduler if provided
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'update_with_metrics') and loss is not None:
                self.scheduler.update_with_metrics(loss, accuracy or 0.0, self.trackers)
            else:
                self.scheduler.step()

        # Determine if this is a mechanism check step
        is_check_step = (self.step_count % self.intervention_frequency == 0)

        # Check for mechanism points on special steps
        if is_check_step:
            # Find mechanism points
            mechanism_points = self.mechanism_grabber.find_mechanism_points(
                self.model, self.trackers, self.metrics
            )

            # Store mechanism points in metrics
            if 'mechanism_points' in self.metrics:
                self.metrics['mechanism_points'].append(mechanism_points)

            # Apply intervention if strong mechanism point found
            if mechanism_points and mechanism_points[0][1] > self.mechanism_threshold:
                layer_name, strength, intervention = mechanism_points[0]
                self.mechanism_grabber.apply_intervention(self.model, intervention)

                # Record intervention
                intervention_record = {
                    'step': self.step_count,
                    'layer': layer_name,
                    'strength': strength,
                    'type': intervention['type'],
                    'loss': loss,
                    'accuracy': accuracy
                }
                self.intervention_history.append(intervention_record)

                # Record in metrics
                if 'interventions' in self.metrics:
                    self.metrics['interventions'].append(
                        (self.step_count, layer_name, strength, intervention['type'])
                    )

        # Update hexagram state if tracker available
        if 'hexagram_state' in self.trackers:
            hex_tracker = self.trackers['hexagram_state']
            if hasattr(hex_tracker, 'get_current_hexagram_index') and hasattr(hex_tracker, 'get_current_hexagram_name'):
                hex_idx = hex_tracker.get_current_hexagram_index()
                hex_name = hex_tracker.get_current_hexagram_name()
                if 'hexagram_states' in self.metrics:
                    self.metrics['hexagram_states'].append((hex_idx, hex_name))

        # Update Wu Xing state if tracker available
        if 'wu_xing_balance' in self.trackers:
            wu_xing_tracker = self.trackers['wu_xing_balance']
            if hasattr(wu_xing_tracker, 'get_element_strengths'):
                element_strengths = wu_xing_tracker.get_element_strengths()
                if 'element_states' in self.metrics:
                    self.metrics['element_states'].append(element_strengths)

        return inner_loss or loss

    def get_intervention_history(self):
        """Get the history of mechanism interventions"""
        return self.intervention_history

    def get_metrics(self):
        """Get optimization metrics"""
        return self.metrics

    # Add parameter group support to match optimizer interface
    @property
    def param_groups(self):
        """Access to parameter groups for compatibility"""
        return self.base_optimizer.param_groups


class SimpleModel(nn.Module):
    """A simple neural network for demonstration"""

    def __init__(self, input_size=9, hidden_sizes=[64, 32, 16], output_size=1):
        super(SimpleModel, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # Input layer
        x = F.relu(self.input_layer(x))

        # Hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Output layer
        return self.output_layer(x)


def generate_data(n_samples=1000, n_features=9):
    """Generate synthetic data for testing"""
    # Create classification problem with specified features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(1, n_features // 2),
        n_redundant=max(1, n_features // 4),
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


def train_epoch(model, optimizer, X, y, batch_size=32, trackers=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Create data loader
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Track metrics for this epoch
    epoch_metrics = {
        'batch_losses': [],
        'gradients': [],
        'weight_norms': []
    }

    # Train on batches
    for batch_X, batch_y in dataloader:
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)

        # Calculate loss
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()

        # Track gradient norms
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item()
        epoch_metrics['gradients'].append(grad_norm)

        # Track weight norms
        weight_norm = 0.0
        for name, param in model.named_parameters():
            weight_norm += param.norm().item()
        epoch_metrics['weight_norms'].append(weight_norm)

        # Update weights with optimizer
        current_loss = loss.item()
        epoch_metrics['batch_losses'].append(current_loss)

        # Update activation statistics for trackers
        activation_mean = outputs.mean().item()
        activation_std = outputs.std().item()

        # Update trackers before optimizer step
        if trackers:
            for name, tracker in trackers.items():
                if hasattr(tracker, 'update_batch'):
                    network_metrics = {
                        'loss': current_loss,
                        'accuracy': (torch.sigmoid(outputs) > 0.5).float().eq(batch_y).float().mean().item(),
                        'loss_decreasing': len(epoch_metrics['batch_losses']) > 1 and
                                           epoch_metrics['batch_losses'][-1] < epoch_metrics['batch_losses'][-2],
                        'gradient_norm': grad_norm,
                        'activation_mean': activation_mean,
                        'activation_std': activation_std,
                        'weight_norm': weight_norm
                    }
                    tracker.update_batch(model, network_metrics)

        # Calculate predictions and accuracy
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        batch_correct = (predicted == batch_y).sum().item()
        batch_total = batch_y.size(0)

        # Update metrics
        total_loss += current_loss * batch_X.size(0)
        correct += batch_correct
        total += batch_total

        # Step the optimizer
        optimizer.step(loss=current_loss, accuracy=batch_correct / batch_total)

    # Calculate average loss and accuracy
    avg_loss = total_loss / total
    accuracy = correct / total

    # Update trackers with epoch results
    if trackers:
        for name, tracker in trackers.items():
            if hasattr(tracker, 'update_epoch'):
                network_metrics = {
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'loss_decreasing': True,  # Placeholder for epoch level
                    'gradient_norm': sum(epoch_metrics['gradients']) / len(epoch_metrics['gradients']),
                    'activation_mean': 0.5,  # Placeholder for epoch level
                    'activation_std': 0.5,  # Placeholder for epoch level
                    'weight_norm': sum(epoch_metrics['weight_norms']) / len(epoch_metrics['weight_norms'])
                }
                tracker.update_epoch(model, network_metrics)

    return avg_loss, accuracy


def validate(model, X, y):
    """Evaluate the model"""
    model.eval()

    with torch.no_grad():
        # Forward pass
        outputs = model(X)

        # Calculate loss
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, y)

        # Calculate accuracy
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        correct = (predicted == y).sum().item()
        total = y.size(0)
        accuracy = correct / total

    return loss.item(), accuracy


def visualize_training(metrics):
    """Visualize training metrics"""
    # Create a figure with multiple plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot losses
    axs[0, 0].plot(metrics['train_loss'], label='Train Loss')
    axs[0, 0].plot(metrics['val_loss'], label='Validation Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss Curves')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Plot accuracies
    axs[0, 1].plot(metrics['train_accuracy'], label='Train Accuracy')
    axs[0, 1].plot(metrics['val_accuracy'], label='Validation Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_title('Accuracy Curves')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Plot learning rates
    if 'learning_rates' in metrics:
        axs[1, 0].plot(metrics['learning_rates'])
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Learning Rate')
        axs[1, 0].set_title('Learning Rate Schedule')
        axs[1, 0].grid(True, alpha=0.3)

    # Plot interventions
    if 'interventions' in metrics and metrics['interventions']:
        # Extract intervention data
        steps = [i[0] for i in metrics['interventions']]
        strengths = [i[2] for i in metrics['interventions']]
        types = [i[3] for i in metrics['interventions']]

        # Create scatter plot
        scatter = axs[1, 1].scatter(steps, strengths, c=range(len(steps)), cmap='viridis', alpha=0.7)

        # Add type labels
        for i, (step, strength, type_name) in enumerate(zip(steps, strengths, types)):
            if i % 2 == 0:  # Skip some labels to avoid clutter
                axs[1, 1].annotate(type_name, (step, strength),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, rotation=45)

        # Add intervention markers to loss plot
        for step in steps:
            if step < len(metrics['train_loss']):
                axs[0, 0].axvline(x=step, color='red', linestyle='--', alpha=0.3)

        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Mechanism Strength')
        axs[1, 1].set_title('Mechanism Interventions')
        axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating DaoML optimization"""
    print("=== DaoML Framework Optimization Example ===")

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X_train, y_train, X_test, y_test = generate_data(n_samples=3000, n_features=9)
    print(
        f"Data shapes: X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}, y_test {y_test.shape}")

    # Create model
    print("\nCreating model...")
    model = SimpleModel(input_size=9, hidden_sizes=[64, 32, 16], output_size=1)
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Initialize I-Ching and Wu Xing trackers
    print("\nInitializing trackers...")
    trackers = {
        'hexagram_state': IChingStateTracker(),
        'wu_xing_balance': WuXingSystem()
    }

    # Initialize mechanism grabber
    print("\nInitializing mechanism grabber...")
    mechanism_grabber = EnhancedMechanismGrabber()

    # FIXED: Create proper optimization setup
    print("\nSetting up optimization approach...")

    # Use list() to ensure parameters aren't consumed by iterator
    params_list = list(model.parameters())

    # Create base optimizer
    base_optimizer = WuWeiOptimizer(params_list, lr=0.01, threshold=0.01, wu_wei_ratio=0.7)

    # Create scheduler
    lr_scheduler = CyclicalElementScheduler(base_optimizer)

    # Create the dual-layer wrapper
    optimizer = DualLayerMechanismOptimizerWrapper(
        model=model,
        base_optimizer=base_optimizer,  # Use the existing optimizer
        mechanism_grabber=mechanism_grabber,
        intervention_frequency=10,
        mechanism_threshold=1.5,
        scheduler=lr_scheduler
    )
    optimizer.register_trackers(trackers)

    # Training settings
    epochs = 50  # Reduced for example
    batch_size = 32

    # Initialize metrics
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'learning_rates': [],
        'interventions': []
    }

    # Training loop
    print("\nStarting training...")
    for epoch in range(epochs):
        # Train one epoch
        train_loss, train_acc = train_epoch(
            model, optimizer, X_train, y_train,
            batch_size=batch_size, trackers=trackers
        )

        # Validate
        val_loss, val_acc = validate(model, X_test, y_test)

        # Record metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_accuracy'].append(train_acc)
        metrics['val_accuracy'].append(val_acc)

        # Record learning rate
        if isinstance(optimizer, DualLayerMechanismOptimizerWrapper):
            lr = optimizer.base_optimizer.param_groups[0]['lr']
            metrics['learning_rates'].append(lr)

            # Get interventions from optimizer
            if hasattr(optimizer, 'get_metrics') and 'interventions' in optimizer.get_metrics():
                metrics['interventions'] = optimizer.get_metrics()['interventions']
        else:
            lr = optimizer.param_groups[0]['lr']
            metrics['learning_rates'].append(lr)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            hexagram_info = ""
            if 'hexagram_state' in trackers:
                hex_idx = trackers['hexagram_state'].get_current_hexagram_index()
                hex_name = trackers['hexagram_state'].get_current_hexagram_name()
                hexagram_info = f", Hexagram: {hex_idx} ({hex_name})"

            element_info = ""
            if 'wu_xing_balance' in trackers:
                dominant = max(trackers['wu_xing_balance'].state.items(), key=lambda x: x[1])[0]
                element_info = f", Element: {dominant.name}"

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {lr:.6f}"
                  f"{hexagram_info}{element_info}")

    print("\nTraining complete.")

    # Visualize results
    print("\nVisualizing training results...")
    visualize_training(metrics)

    # Evaluate final model
    print("\nFinal evaluation:")
    final_val_loss, final_val_acc = validate(model, X_test, y_test)
    print(f"Final validation loss: {final_val_loss:.4f}, accuracy: {final_val_acc:.4f}")

    # Print optimizer statistics
    if isinstance(optimizer, DualLayerMechanismOptimizerWrapper):
        interventions = optimizer.get_intervention_history()
        print(f"\nTotal mechanism interventions: {len(interventions)}")

        # Count interventions by type
        intervention_types = {}
        for intervention in interventions:
            int_type = intervention['type']
            intervention_types[int_type] = intervention_types.get(int_type, 0) + 1

        print("Intervention types:")
        for int_type, count in intervention_types.items():
            print(f"  - {int_type}: {count}")

    # Print hexagram transition statistics
    if 'hexagram_state' in trackers:
        hex_history = trackers['hexagram_state'].hexagram_transition_history
        print(f"\nTotal hexagram transitions: {len(hex_history)}")

        # Most frequent hexagrams
        hexagram_counts = {}
        for from_hex, to_hex in hex_history:
            hexagram_counts[to_hex] = hexagram_counts.get(to_hex, 0) + 1

        print("Most frequent hexagrams:")
        for hex_idx, count in sorted(hexagram_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            hex_name = trackers['hexagram_state'].hexagrams.hexagram_names.get(hex_idx, f"Hexagram {hex_idx}")
            print(f"  - {hex_idx} ({hex_name}): {count}")

    # Print Wu Xing element statistics
    if 'wu_xing_balance' in trackers:
        print("\nFinal Wu Xing element strengths:")
        for element, strength in trackers['wu_xing_balance'].state.items():
            print(f"  - {element.name}: {strength:.2f}")

    print("\n=== Example complete ===")


if __name__ == "__main__":
    main()
