import math

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


#################################################
# I-Ching Fundamentals
#################################################

# Generate the Eight Trigrams (Bagua)
class Bagua:
    def __init__(self):
        # Trigram definitions (binary representation: 0=yin, 1=yang)
        self.trigrams = {
            'qian': [1, 1, 1],  # ☰ Heaven
            'kun': [0, 0, 0],  # ☷ Earth
            'zhen': [0, 0, 1],  # ☳ Thunder
            'gen': [1, 0, 0],  # ☶ Mountain
            'kan': [0, 1, 0],  # ☵ Water
            'li': [1, 0, 1],  # ☲ Fire
            'xun': [1, 1, 0],  # ☴ Wind
            'dui': [0, 1, 1]  # ☱ Lake
        }

        # Create reverse mapping (binary to name)
        self.trigram_names = {}
        for name, trigram in self.trigrams.items():
            binary = ''.join(map(str, trigram))
            self.trigram_names[binary] = name

        # Define element associations for each trigram
        self.trigram_elements = {
            'qian': 'metal',  # Heaven - Metal
            'kun': 'earth',  # Earth - Earth
            'zhen': 'wood',  # Thunder - Wood
            'gen': 'earth',  # Mountain - Earth
            'kan': 'water',  # Water - Water
            'li': 'fire',  # Fire - Fire
            'xun': 'wood',  # Wind - Wood
            'dui': 'metal'  # Lake - Metal
        }

        # Define the Prior Heaven (Fu Xi) sequence
        self.prior_heaven_sequence = [
            'kun', 'gen', 'kan', 'xun',
            'zhen', 'li', 'dui', 'qian'
        ]

        # Define the Later Heaven (King Wen) sequence
        self.later_heaven_sequence = [
            'kan', 'kun', 'zhen', 'gen',
            'qian', 'dui', 'li', 'xun'
        ]


# Generate the 64 Hexagrams
class Hexagrams:
    def __init__(self):
        self.bagua = Bagua()
        self.hexagrams = self._generate_hexagrams()
        self.hexagram_names = self._get_hexagram_names()
        self.change_lines = self._calculate_change_lines()
        self.opposite_hexagrams = self._calculate_opposites()
        self.nuclear_hexagrams = self._calculate_nuclear_hexagrams()

    def _generate_hexagrams(self):
        """Generate all 64 hexagrams as combinations of two trigrams"""
        hexagrams = {}
        trigram_values = list(self.bagua.trigrams.values())

        # Enumerate all 64 combinations
        for i, upper in enumerate(trigram_values):
            for j, lower in enumerate(trigram_values):
                # Combine upper and lower trigrams to form hexagram
                lines = upper + lower
                # Convert to integer index (0-63)
                idx = sum(line * (2 ** i) for i, line in enumerate(reversed(lines)))
                hexagrams[idx] = {
                    'lines': lines,
                    'upper': upper,
                    'lower': lower,
                    'binary': ''.join(map(str, lines))
                }

        return hexagrams

    def _get_hexagram_names(self):
        """Get traditional names for the 64 hexagrams according to the King Wen sequence"""
        # Traditional names with their numerical designations
        names = {
            0: "Kun (地) - The Receptive",  # ䷁ Earth
            1: "Gou (姤) - Coming to Meet",  # ䷫ Wind over Heaven
            2: "Dun (遯) - Retreat",  # ䷠ Mountain over Heaven
            3: "Pi (否) - Standstill",  # ䷋ Earth over Heaven
            4: "Guan (觀) - Contemplation",  # ䷓ Earth over Wind
            5: "Bo (剝) - Splitting Apart",  # ䷖ Mountain over Earth
            6: "Gen (艮) - Keeping Still",  # ䷳ Mountain
            7: "Sui (隨) - Following",  # ䷐ Lake over Thunder
            8: "Gu (蠱) - Work on the Decayed",  # ䷑ Mountain over Wind
            9: "Lin (臨) - Approach",  # ䷒ Earth over Lake
            10: "Tai (泰) - Peace",  # ䷊ Heaven over Earth
            11: "Fou (否) - Abundance",  # ䷙ Fire over Heaven
            12: "Tong Ren (同人) - Fellowship",  # ䷌ Heaven over Fire
            13: "Da You (大有) - Great Possession",  # ䷍ Fire over Heaven
            14: "Shi (師) - The Army",  # ䷆ Earth over Water
            15: "Bi (比) - Holding Together",  # ䷇ Water over Earth
            16: "Xian (咸) - Influence",  # ䷞ Lake over Mountain
            17: "Heng (恆) - Duration",  # ䷟ Thunder over Wind
            18: "Dui (兌) - The Joyous",  # ䷹ Lake
            19: "Guai (夬) - Breakthrough",  # ䷪ Lake over Heaven
            20: "Qian (乾) - The Creative",  # ䷀ Heaven
            21: "Sheng (升) - Pushing Upward",  # ䷭ Earth over Wood
            22: "Kun (困) - Oppression",  # ䷮ Water over Earth
            23: "Jing (井) - The Well",  # ䷯ Wind over Water
            24: "Meng (蒙) - Youthful Folly",  # ䷃ Mountain over Water
            25: "Huan (渙) - Dispersion",  # ䷲ Wind over Water
            26: "Lu (履) - Treading",  # ䷉ Heaven over Lake
            27: "Xun (巽) - The Gentle",  # ䷸ Wind
            28: "Zhongfu (中孚) - Inner Truth",  # ䷼ Wind over Lake
            29: "Kan (坎) - The Abysmal",  # ䷜ Water
            30: "Li (離) - The Clinging",  # ䷝ Fire
            31: "Xing (革) - Revolution",  # ䷰ Lake over Fire
            32: "Da Guo (大過) - Preponderance",  # ䷛ Lake over Mountain
            33: "Yi (頤) - The Corners of the Mouth",  # ䷚ Mountain over Thunder
            34: "Da Chuang (大壯) - Great Power",  # ䷡ Thunder over Heaven
            35: "Jin (晉) - Progress",  # ䷢ Fire over Earth
            36: "Ming Yi (明夷) - Darkening of Light",  # ䷣ Earth over Fire
            37: "Jia Ren (家人) - The Family",  # ䷤ Wind over Fire
            38: "Kui (睽) - Opposition",  # ䷥ Fire over Lake
            39: "Jian (蹇) - Obstruction",  # ䷦ Water over Mountain
            40: "Jie (解) - Deliverance",  # ䷧ Thunder over Water
            41: "Sun (損) - Decrease",  # ䷨ Mountain over Lake
            42: "Yi (益) - Increase",  # ䷩ Wind over Thunder
            43: "Gou (姤) - Coming to Meet",  # ䷫ Heaven over Wind
            44: "Cui (萃) - Gathering Together",  # ䷬ Lake over Earth
            45: "Xiao Guo (小過) - Small Excess",  # ䷽ Mountain over Thunder
            46: "Ji Ji (既濟) - After Completion",  # ䷾ Fire over Water
            47: "Wei Ji (未濟) - Before Completion",  # ䷿ Water over Fire
            48: "Xu (需) - Waiting",  # ䷄ Water over Heaven
            49: "Song (訟) - Conflict",  # ䷅ Heaven over Water
            50: "Huan (渙) - Dispersion",  # ䷲ Water over Wind
            51: "Zhen (震) - The Arousing",  # ䷲ Thunder
            52: "Feng (豐) - Abundance",  # ䷶ Thunder over Fire
            53: "Lü (旅) - The Wanderer",  # ䷷ Fire over Mountain
            54: "Sun (巽) - The Gentle",  # ䷸ Wind over Mountain
            55: "Tun (屯) - Difficulty at the Beginning",  # ䷂ Water over Thunder
            56: "Ben (賁) - Grace",  # ䷕ Fire over Mountain
            57: "Qian (謙) - Modesty",  # ䷎ Mountain over Earth
            58: "Yu (豫) - Enthusiasm",  # ䷏ Thunder over Earth
            59: "Guai (夬) - Breakthrough",  # ䷪ Heaven over Lake
            60: "Fu (復) - Return",  # ䷗ Thunder over Earth
            61: "Wu Wang (無妄) - Innocence",  # ䷘ Thunder over Heaven
            62: "Da Xu (大畜) - Great Taming",  # ䷙ Mountain over Heaven
            63: "Qian (乾) - The Creative"  # ䷀ Heaven
        }

        return names

    def _calculate_change_lines(self):
        """Calculate transitions between hexagrams based on line changes"""
        changes = {}
        for i in range(64):
            changes[i] = {}
            for j in range(64):
                if i == j:
                    continue
                # Calculate Hamming distance (number of changed lines)
                lines_i = self.hexagrams[i]['lines']
                lines_j = self.hexagrams[j]['lines']
                changed_lines = [k for k in range(6) if lines_i[k] != lines_j[k]]
                changes[i][j] = changed_lines
        return changes

    def _calculate_opposites(self):
        """Calculate opposite hexagrams (all lines reversed)"""
        opposites = {}
        for i in range(64):
            lines = self.hexagrams[i]['lines']
            # Reverse all lines (0->1, 1->0)
            reversed_lines = [1 - line for line in lines]
            # Convert to index
            idx = sum(line * (2 ** i) for i, line in enumerate(reversed(reversed_lines)))
            opposites[i] = idx
        return opposites

    def _calculate_nuclear_hexagrams(self):
        """Calculate nuclear hexagrams (derived from the inner four lines)"""
        nuclear = {}
        for i in range(64):
            lines = self.hexagrams[i]['lines']
            # Upper nuclear trigram is formed from lines 2, 3, 4
            upper_nuclear = lines[1:4]
            # Lower nuclear trigram is formed from lines 3, 4, 5
            lower_nuclear = lines[2:5]
            # Combine to form nuclear hexagram
            nuclear_lines = upper_nuclear + lower_nuclear
            # Convert to index
            idx = sum(line * (2 ** i) for i, line in enumerate(reversed(nuclear_lines)))
            nuclear[i] = idx
        return nuclear

    def get_hexagram_transition_type(self, from_hex, to_hex):
        """Determine the type of transition between hexagrams"""
        if from_hex == to_hex:
            return "identical"

        changed_lines = self.change_lines[from_hex][to_hex]

        if len(changed_lines) == 1:
            return "single_line"
        elif len(changed_lines) == 6:
            return "opposite"
        elif to_hex == self.nuclear_hexagrams[from_hex]:
            return "nuclear"
        elif len(changed_lines) == 3 and set(changed_lines) == {0, 2, 4} or set(changed_lines) == {1, 3, 5}:
            return "alternating"
        else:
            return "standard"


#################################################
# I-Ching Based Initialization and Operations
#################################################

class YinYangInitializer:
    """Initialize parameters using balanced Yin-Yang principles"""

    def __init__(self, hexagram_idx=1):
        self.hexagrams = Hexagrams()
        self.hex_idx = hexagram_idx
        self.lines = self.hexagrams.hexagrams[hexagram_idx]['lines']

    def initialize_tensor(self, tensor):
        """Initialize a tensor based on hexagram structure"""
        fan_in, fan_out = self._calculate_fan(tensor)

        # Base scale determined by hexagram structure
        yin_count = 6 - sum(self.lines)  # Count of yin lines (0s)
        yang_count = sum(self.lines)  # Count of yang lines (1s)

        # Calculate balance factors
        yin_factor = yin_count / 6.0
        yang_factor = yang_count / 6.0

        # Calculate adaptive scales
        yin_scale = math.sqrt(6.0 / fan_in) * yin_factor
        yang_scale = math.sqrt(6.0 / fan_out) * yang_factor

        # Create balanced initialization pattern
        with torch.no_grad():
            # Generate the base random values
            tensor.uniform_(-1, 1)

            # Generate yin-yang pattern based on hexagram
            pattern = self._create_hexagram_pattern(tensor.size())

            # Apply pattern-based scaling
            yin_mask = 1.0 - pattern
            yang_mask = pattern

            yin_component = tensor * yin_mask * yin_scale
            yang_component = tensor * yang_mask * yang_scale

            # Combine components
            tensor.copy_(yin_component + yang_component)

        return tensor

    def _calculate_fan(self, tensor):
        """Calculate fan_in and fan_out for initialization scaling, handling all tensor dimensions"""
        dimensions = tensor.dim()

        if dimensions == 1:  # Bias vector
            fan_in = 1
            fan_out = tensor.size(0)
        elif dimensions == 2:  # Linear layer weights
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:  # Convolution layers etc.
            receptive_field_size = 1
            for s in tensor.size()[2:]:
                receptive_field_size *= s
            fan_in = tensor.size(1) * receptive_field_size
            fan_out = tensor.size(0) * receptive_field_size

        return fan_in, fan_out

    def _create_hexagram_pattern(self, size):
        """Create a pattern based on hexagram structure that matches tensor dimensions"""
        # Create base pattern from hexagram lines
        pattern = torch.tensor(self.lines, dtype=torch.float32)

        # Repeat pattern to cover tensor
        flat_len = np.prod(size)
        repeats = math.ceil(flat_len / 6)
        expanded = pattern.repeat(repeats)

        # Reshape to tensor dimensions
        return expanded[:flat_len].reshape(size)

    def set_hexagram(self, hex_idx):
        """Update the hexagram used for initialization"""
        self.hex_idx = hex_idx
        self.lines = self.hexagrams.hexagrams[hex_idx]['lines']


class HexagramTransformation:
    """Implement transformations between hexagrams as neural operations"""

    def __init__(self):
        self.hexagrams = Hexagrams()

    def create_transformation_matrix(self, from_hex, to_hex, size):
        """Generate a transformation matrix based on hexagram transition"""
        transition_type = self.hexagrams.get_hexagram_transition_type(from_hex, to_hex)

        # Create appropriate transformation based on transition type
        if transition_type == "identical":
            return torch.eye(size)
        elif transition_type == "single_line":
            return self._create_single_line_transformation(from_hex, to_hex, size)
        elif transition_type == "opposite":
            return self._create_opposite_transformation(size)
        elif transition_type == "nuclear":
            return self._create_nuclear_transformation(from_hex, to_hex, size)
        elif transition_type == "alternating":
            return self._create_alternating_transformation(from_hex, to_hex, size)
        else:
            return self._create_standard_transformation(from_hex, to_hex, size)

    def _create_single_line_transformation(self, from_hex, to_hex, size):
        """Create transformation for single line change (subtle, focused change)"""
        # Start with identity matrix
        matrix = torch.eye(size)

        # Find which line changed
        changed_line = self.hexagrams.change_lines[from_hex][to_hex][0]

        # Apply line-specific transformation
        # - A low line change affects the beginning of the matrix
        # - A high line change affects the end of the matrix
        line_factor = (changed_line + 1) / 7.0  # Scale to 1/7 through 6/7

        # Apply graduated transformation based on position
        positions = torch.arange(size, dtype=torch.float32) / size
        influence = torch.exp(-10 * (positions - line_factor) ** 2)  # Gaussian centered at line_factor

        # Apply influence to matrix diagonal
        diag_influence = 0.8 + 0.4 * influence  # Scale to range [0.8, 1.2]
        matrix = matrix * diag_influence.unsqueeze(1)

        return matrix

    def _create_opposite_transformation(self, size):
        """Create transformation for opposite hexagram (complete reversal)"""
        # Create a matrix that reverses the representation
        # This is effectively a reflection matrix
        matrix = torch.zeros(size, size)

        # Fill in reflection pattern
        for i in range(size):
            matrix[i, size - 1 - i] = 1.0

        return matrix

    def _create_nuclear_transformation(self, from_hex, to_hex, size):
        """Create transformation for nuclear hexagram (inner essence)"""
        # Create a matrix that emphasizes the central elements
        matrix = torch.eye(size)

        # Create a central emphasis pattern
        center = size // 2
        positions = torch.arange(size, dtype=torch.float32)
        distance_from_center = torch.abs(positions - center) / center
        central_emphasis = 1.0 + 0.5 * (1.0 - distance_from_center)

        # Apply the pattern to the diagonal
        for i in range(size):
            matrix[i, i] = central_emphasis[i]

        return matrix

    def _create_alternating_transformation(self, from_hex, to_hex, size):
        """Create transformation for alternating lines (rhythmic pattern)"""
        # Create an alternating emphasis pattern
        matrix = torch.eye(size)

        # Apply alternating pattern
        for i in range(size):
            if i % 2 == 0:
                matrix[i, i] = 1.2  # Enhance even positions
            else:
                matrix[i, i] = 0.8  # Reduce odd positions

        return matrix

    def _create_standard_transformation(self, from_hex, to_hex, size):
        """Create transformation for standard multi-line changes"""
        # Get the changed lines
        changed_lines = self.hexagrams.change_lines[from_hex][to_hex]
        num_changes = len(changed_lines)

        # Create transformation matrix
        matrix = torch.eye(size)

        # Apply scaled transformations based on change ratio
        change_ratio = num_changes / 6.0

        # Apply changes proportional to number of line changes
        for i in range(size):
            position_ratio = i / size
            # Apply different transformations to different parts of the matrix
            # based on position and change pattern
            if abs(position_ratio - change_ratio) < 0.3:
                # Areas corresponding to changed lines get enhanced
                matrix[i, i] = 1.2
            else:
                # Other areas slightly reduced
                matrix[i, i] = 0.9

        return matrix


class BaguaActivations(nn.Module):
    """Activation functions based on the eight trigrams"""

    def __init__(self):
        super(BaguaActivations, self).__init__()
        self.bagua = Bagua()

    def forward(self, x, trigram_name):
        """Apply trigram-specific activation"""
        if trigram_name == 'qian':  # Heaven - Creative, strong
            return F.elu(x) + 0.5 * torch.tanh(x)
        elif trigram_name == 'kun':  # Earth - Receptive, nurturing
            return torch.sigmoid(x)
        elif trigram_name == 'zhen':  # Thunder - Arousing, movement
            return F.leaky_relu(x, 0.2)
        elif trigram_name == 'gen':  # Mountain - Keeping still, stability
            # GELU with slight clamping for stability
            return F.gelu(torch.clamp(x, -5.0, 5.0))
        elif trigram_name == 'kan':  # Water - Abysmal, flowing
            return torch.tanh(x)
        elif trigram_name == 'li':  # Fire - Clinging, clarity
            # Modified SiLU (Swish) for feature clarity
            return x * torch.sigmoid(1.2 * x)
        elif trigram_name == 'xun':  # Wind - Gentle, penetrating
            # Softer variant of ReLU
            return F.softplus(x)
        elif trigram_name == 'dui':  # Lake - Joyous, open
            # Modified ELU with enhanced positive values
            return F.elu(x) + 0.2 * x
        else:
            # Default to ReLU
            return F.relu(x)

    def get_all_activations(self, x):
        """Apply all eight trigram activations and return results"""
        results = {}
        for trigram in self.bagua.trigrams.keys():
            results[trigram] = self.forward(x, trigram)
        return results


class IChingStateTracker(nn.Module):
    """Track the hexagram state of the network during training"""

    def __init__(self):
        super(IChingStateTracker, self).__init__()
        self.hexagrams = Hexagrams()
        self.register_buffer('current_hexagram', torch.zeros(64))
        # Initialize to Qian hexagram (all yang lines)
        self.current_hexagram[63] = 1.0
        self.hexagram_transition_history = []

    def update_state(self, network_metrics):
        """Update the hexagram state based on network conditions"""
        # Extract network metrics
        loss = network_metrics['loss']
        accuracy = network_metrics['accuracy']
        gradient_norm = network_metrics['gradient_norm']
        activation_mean = network_metrics['activation_mean']
        activation_std = network_metrics['activation_std']
        weight_norm = network_metrics['weight_norm']

        # Calculate six lines based on network metrics
        lines = [
            # Line 1 (bottom): Loss trend (yang if decreasing, yin if increasing)
            1 if network_metrics['loss_decreasing'] else 0,

            # Line 2: Accuracy (yang if above threshold, yin if below)
            1 if accuracy > 0.75 else 0,

            # Line 3: Gradient stability (yang if stable, yin if erratic)
            1 if gradient_norm < 1.0 else 0,

            # Line 4: Activation mean (yang if positive bias, yin if negative)
            1 if activation_mean > 0 else 0,

            # Line 5: Activation distribution (yang if varied, yin if uniform)
            1 if activation_std > 0.5 else 0,

            # Line 6 (top): Weight norm (yang if moderate, yin if too large or small)
            1 if 0.5 < weight_norm < 5.0 else 0
        ]

        # Convert lines to hexagram index (0-63)
        hex_index = sum(line * (2 ** i) for i, line in enumerate(reversed(lines)))

        # Record transition
        prev_hex = torch.argmax(self.current_hexagram).item()
        if prev_hex != hex_index:
            self.hexagram_transition_history.append((prev_hex, hex_index))

        # Update current hexagram (one-hot encoding)
        self.current_hexagram.zero_()
        self.current_hexagram[hex_index] = 1.0

        return hex_index

    def get_transition_type(self):
        """Get the type of the most recent transition"""
        if len(self.hexagram_transition_history) == 0:
            return "initial"

        from_hex, to_hex = self.hexagram_transition_history[-1]
        return self.hexagrams.get_hexagram_transition_type(from_hex, to_hex)

    def get_current_hexagram_lines(self):
        """Get the current hexagram's lines"""
        hex_index = torch.argmax(self.current_hexagram).item()
        return self.hexagrams.hexagrams[hex_index]['lines']

    def get_current_hexagram_name(self):
        """Get the current hexagram's name"""
        hex_index = torch.argmax(self.current_hexagram).item()
        return self.hexagrams.hexagram_names[hex_index]

    def get_current_hexagram_index(self):
        """Get the current hexagram's index"""
        return torch.argmax(self.current_hexagram).item()

    def update_batch(self, model, network_metrics):
        """
        Update the hexagram state based on batch metrics

        Args:
            model: The neural network model
            network_metrics: Dictionary of network metrics
        """
        self.update_state(network_metrics)

    def update_epoch(self, model, network_metrics):
        """
        Update the hexagram state based on epoch metrics

        Args:
            model: The neural network model
            network_metrics: Dictionary of network metrics
        """
        self.update_state(network_metrics)

    # Add this to the WuXingSystem class if it doesn't exist
