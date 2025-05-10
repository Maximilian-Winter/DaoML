import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, List, Tuple, Optional
import math
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


#################################################
# Lo Shu Magic Square - The Perfect Square
#################################################

class LoShuTensor:
    """
    Lo Shu Magic Square implementation with Wu Xing elemental qualities.

    The Lo Shu is a 3x3 magic square where all rows, columns, and diagonals sum to 15.
    In Chinese cosmology, it represents perfect mathematical balance and harmony.
    """

    def __init__(self):
        # Traditional Lo Shu square
        self.lo_shu = torch.tensor([
            [4, 9, 2],
            [3, 5, 7],
            [8, 1, 6]
        ]).float()

        # Normalize to range [0.5, 1.0] to avoid zeroing out connections
        self.normalized = 0.5 + 0.5 * (self.lo_shu - self.lo_shu.min()) / (self.lo_shu.max() - self.lo_shu.min())

        # Element mapping in Lo Shu:
        # 1 (Water), 2 (Earth), 3 (Wood), 4 (Wood), 5 (Earth),
        # 6 (Water), 7 (Metal), 8 (Metal), 9 (Fire)
        self.element_mapping = {
            1: 'water', 2: 'earth', 3: 'wood', 4: 'wood', 5: 'earth',
            6: 'water', 7: 'metal', 8: 'metal', 9: 'fire'
        }

    def enhance_with_wu_xing(self, element_strengths=None):
        """Enhance Lo Shu with Wu Xing elemental qualities"""
        if element_strengths is None:
            # Default element strength modifiers
            element_strengths = {
                'water': 1.2,  # Adaptive, flowing
                'wood': 1.1,  # Growth, expansion
                'fire': 1.3,  # Transformation
                'earth': 0.9,  # Stability
                'metal': 0.8  # Precision, contraction
            }

        # Create element mask tensor with same shape as lo_shu
        element_mask = torch.ones_like(self.lo_shu)

        # Apply element strengths to corresponding positions
        for i in range(3):
            for j in range(3):
                value = self.lo_shu[i, j].item()
                element = self.element_mapping[value]
                element_mask[i, j] = element_strengths[element]

        # Apply element mask to normalized lo_shu
        enhanced = self.normalized * element_mask

        # Re-normalize to preserve energy
        enhanced = enhanced / enhanced.mean() * self.normalized.mean()

        return enhanced

    def get_weight_template(self, in_features, out_features):
        """Create a weight template based on Lo Shu pattern"""
        # Enhanced Lo Shu with Wu Xing qualities
        enhanced_lo_shu = self.enhance_with_wu_xing()

        # Create template matching input size
        template = torch.ones(out_features, in_features)

        # Tile Lo Shu pattern to fill template
        for i in range(0, out_features, 3):
            for j in range(0, in_features, 3):
                # Get bounds respecting matrix dimensions
                i_end = min(i + 3, out_features)
                j_end = min(j + 3, in_features)

                # Fill the block with Lo Shu pattern (partial if at edges)
                template[i:i_end, j:j_end] = enhanced_lo_shu[:i_end - i, :j_end - j]

        return template


#################################################
# Bagua and I-Ching Fundamentals
#################################################

class Bagua:
    """The Eight Trigrams (Bagua) representing fundamental principles"""

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


class Hexagrams:
    """The 64 Hexagrams of the I-Ching"""

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
        """Get traditional names for the 64 hexagrams"""
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


class HexagramStateTracker(nn.Module):
    """Track the hexagram state of the network during training"""

    def __init__(self):
        super(HexagramStateTracker, self).__init__()
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
        """Update the hexagram state based on batch metrics"""
        self.update_state(network_metrics)

    def update_epoch(self, model, network_metrics):
        """Update the hexagram state based on epoch metrics"""
        self.update_state(network_metrics)


#################################################
# Wu Xing (Five Elements) System
#################################################

class Element(Enum):
    """The Five Elements (Wu Xing)"""
    WATER = 0
    WOOD = 1
    FIRE = 2
    EARTH = 3
    METAL = 4

    def __str__(self):
        return self.name.capitalize()

    @property
    def color(self):
        colors = {
            Element.WATER: 'blue',
            Element.WOOD: 'green',
            Element.FIRE: 'red',
            Element.EARTH: 'brown',
            Element.METAL: 'silver'
        }
        return colors[self]

    @property
    def nature(self):
        natures = {
            Element.WATER: "Descending",
            Element.WOOD: "Seeking Rise",
            Element.FIRE: "Rising",
            Element.EARTH: "Stabilizing",
            Element.METAL: "Seeking Descent"
        }
        return natures[self]


class WuXingSystem:
    """Complete Wu Xing system with all elemental cycles"""

    def __init__(self, initial_state=None):
        """Initialize the system with element values"""
        # Default initial state: balanced
        if initial_state is None:
            initial_state = {element: 20.0 for element in Element}

        self.state = initial_state.copy()
        self.history = [self.state.copy()]

        # Define generation cycle (生, sheng): each element generates the next
        self.generation_cycle = {
            Element.WATER: Element.WOOD,  # Water nourishes Wood
            Element.WOOD: Element.FIRE,  # Wood feeds Fire
            Element.FIRE: Element.EARTH,  # Fire creates Earth (ash)
            Element.EARTH: Element.METAL,  # Earth bears Metal
            Element.METAL: Element.WATER  # Metal collects Water (condensation)
        }

        # Define conquest cycle (克, ke): each element conquers another
        self.conquest_cycle = {
            Element.WATER: Element.FIRE,  # Water extinguishes Fire
            Element.FIRE: Element.METAL,  # Fire melts Metal
            Element.METAL: Element.WOOD,  # Metal cuts Wood
            Element.WOOD: Element.EARTH,  # Wood breaks Earth
            Element.EARTH: Element.WATER  # Earth absorbs Water
        }

        # Coefficients for various cycles
        self.coefficients = {
            'generation': 0.3,  # Generation effect strength
            'conquest': 0.4,  # Conquest effect strength
            'balance': 0.15  # Balance effect strength
        }

    def step(self, dt=0.1):
        """Advance the system by one time step with proper bounds"""
        new_state = self.state.copy()

        # Calculate all interactions
        for source in Element:
            for target in Element:
                if source == target:
                    continue

                # Generation effect (source generates target)
                if self.generation_cycle[source] == target:
                    # Never take more than what's available
                    effect = min(new_state[source] * 0.3, self.coefficients['generation'] * new_state[source] * dt)
                    new_state[target] += effect
                    # Source decreases when generating, but never below minimum
                    new_state[source] = max(0.1, new_state[source] - effect * 0.5)

                # Conquest effect (source conquers target)
                if self.conquest_cycle[source] == target:
                    # Scale conquest effect by available target energy
                    effect = -min(new_state[target] * 0.5, self.coefficients['conquest'] * new_state[source] * dt)
                    new_state[target] = max(0.1, new_state[target] + effect)

        # Apply natural tendencies based on elemental nature with proper bounds
        for element in Element:
            if element == Element.FIRE:  # Fire rises
                new_state[element] *= min(1.05, 1 + 0.01 * dt)  # Limited natural increase
            elif element == Element.WATER:  # Water descends
                new_state[element] = max(0.1, new_state[element] * (1 - 0.01 * dt))  # Prevent negative
            elif element == Element.EARTH:  # Earth stabilizes
                # Move toward the average
                avg = sum(new_state.values()) / len(new_state)
                new_state[element] += (avg - new_state[element]) * 0.1 * dt

        # Hard bounds to prevent extreme values
        for element in Element:
            new_state[element] = max(0.1, min(100.0, new_state[element]))

        # Apply soft normalization to prevent all elements growing together
        total = sum(new_state.values())
        if total > 300:  # If total energy becomes too high
            scaling_factor = 300 / total
            for element in Element:
                new_state[element] *= scaling_factor

        self.state = new_state
        self.history.append(self.state.copy())

        return self.state

    def get_dominant_element(self):
        """Return the currently dominant element"""
        return max(self.state.items(), key=lambda x: x[1])[0]

    def get_element_strengths(self):
        """Get current strengths of all elements"""
        return self.state

    def update_batch(self, model, network_metrics):
        """Update based on batch metrics"""
        # Water (adaptability) strengthened by decreasing loss, varied gradients
        if network_metrics['loss_decreasing']:
            self.state[Element.WATER] += 0.1
        if network_metrics['gradient_norm'] > 0.5:
            self.state[Element.WATER] += 0.05

        # Wood (growth) strengthened by increasing accuracy
        if network_metrics['accuracy'] > 0.7:
            self.state[Element.WOOD] += 0.1

        # Fire (transformation) strengthened by high gradient activity
        if network_metrics['gradient_norm'] > 1.0:
            self.state[Element.FIRE] += 0.15

        # Earth (stability) strengthened by balanced metrics
        if 0.7 < network_metrics['accuracy'] < 0.9:
            self.state[Element.EARTH] += 0.1
        if 0.5 < network_metrics['weight_norm'] < 2.0:
            self.state[Element.EARTH] += 0.05

        # Metal (precision) strengthened by high accuracy, low gradients
        if network_metrics['accuracy'] > 0.9:
            self.state[Element.METAL] += 0.15
        if network_metrics['gradient_norm'] < 0.1:
            self.state[Element.METAL] += 0.05

        # Apply system dynamics
        self.step(dt=0.01)

    def update_epoch(self, model, network_metrics):
        """Update based on epoch metrics"""
        # Similar to update_batch but with larger effects
        if network_metrics['loss_decreasing']:
            self.state[Element.WATER] += 1.0
        if network_metrics['gradient_norm'] > 0.5:
            self.state[Element.WATER] += 0.5

        if network_metrics['accuracy'] > 0.7:
            self.state[Element.WOOD] += 1.0

        if network_metrics['gradient_norm'] > 1.0:
            self.state[Element.FIRE] += 1.5

        if 0.7 < network_metrics['accuracy'] < 0.9:
            self.state[Element.EARTH] += 1.0
        if 0.5 < network_metrics['weight_norm'] < 2.0:
            self.state[Element.EARTH] += 0.5

        if network_metrics['accuracy'] > 0.9:
            self.state[Element.METAL] += 1.5
        if network_metrics['gradient_norm'] < 0.1:
            self.state[Element.METAL] += 0.5

        # Apply system dynamics with a larger time step
        self.step(dt=0.1)


#################################################
# Wu Wei Optimizer - Non-Action (Advanced)
#################################################

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
        """Initialize the enhanced Wu Wei Optimizer"""
        if not isinstance(params, (list, tuple)):
            params = list(params)

        if len(params) == 0:
            raise ValueError("Optimizer got an empty parameter list")

        defaults = dict(lr=lr, betas=betas, threshold=threshold,
                        adapt_rate=adapt_rate, wu_wei_ratio=wu_wei_ratio)

        super(WuWeiOptimizer, self).__init__(params, defaults)

        # Inner optimizer for standard updates (Adam)
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
        """Perform a single optimization step with Wu Wei principles"""
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
        """Enhanced step method that incorporates tracker information"""
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


#################################################
# Celestial Mechanism Grabber
#################################################

class CelestialMechanismGrabber:
    """
    Enhanced implementation of the mechanism-grabbing principle.
    Identifies critical points with minimal computational overhead and maximizes intervention impact.
    """

    def __init__(self):
        """Initialize the celestial mechanism grabber"""
        self.hexagrams = Hexagrams()
        self.wu_xing = WuXingSystem()
        self.bagua_activations = BaguaActivations()

        # Map network states to trigram energies
        self.network_trigram_mapping = {
            'gradient_flow': ['kan', 'li'],  # Water and Fire for gradient flow
            'weight_stability': ['gen', 'kun'],  # Mountain and Earth for stability
            'activation_pattern': ['zhen', 'xun'],  # Thunder and Wind for activation
            'optimization_state': ['qian', 'dui']  # Heaven and Lake for optimization
        }

        # Intervention types
        self.intervention_types = {
            'activate': self._activate_intervention,
            'stabilize': self._stabilize_intervention,
            'balance': self._balance_intervention,
            'transform': self._transform_intervention,
            'harmonize': self._harmonize_intervention
        }

        # Timing windows for interventions (based on cosmic cycles)
        self.timing_cycles = {
            'short': 7,  # 7-step cycle (like days in a week)
            'medium': 28,  # 28-step cycle (like lunar month)
            'long': 60  # 60-step cycle (like sexagenary cycle)
        }

        # Initialize timing counters
        self.step_counter = 0
        self.intervention_history = []

    def analyze_trigram_balance(self, model, metrics):
        """Analyze trigram energy balance in network"""
        trigram_strengths = {trigram: 0.0 for trigram in self.bagua_activations.bagua.trigrams}

        # Calculate trigram energies based on network state
        for aspect, trigrams in self.network_trigram_mapping.items():
            if aspect == 'gradient_flow':
                # Measure gradient smoothness and magnitude
                gradient_norm = metrics.get('gradient_norm', 1.0)
                kan_strength = min(1.0, 1.0 / (1.0 + gradient_norm))  # Water - smooth flow
                li_strength = min(1.0, gradient_norm / 2.0)  # Fire - transformation
                trigram_strengths['kan'] += kan_strength
                trigram_strengths['li'] += li_strength

            elif aspect == 'weight_stability':
                # Measure weight stability
                weight_norm = metrics.get('weight_norm', 1.0)
                loss = metrics.get('loss', 0.5)

                gen_strength = min(1.0, 1.0 / (1.0 + weight_norm))  # Mountain - stillness
                kun_strength = min(1.0, 1.0 / (1.0 + loss))  # Earth - receptivity

                trigram_strengths['gen'] += gen_strength
                trigram_strengths['kun'] += kun_strength

            elif aspect == 'activation_pattern':
                # Measure activation patterns
                activation_mean = metrics.get('activation_mean', 0.0)
                activation_std = metrics.get('activation_std', 0.5)

                zhen_strength = min(1.0, activation_std)  # Thunder - arousing
                xun_strength = min(1.0, 1.0 - abs(activation_mean))  # Wind - gentle

                trigram_strengths['zhen'] += zhen_strength
                trigram_strengths['xun'] += xun_strength

            elif aspect == 'optimization_state':
                # Measure optimization progress
                accuracy = metrics.get('accuracy', 0.5)
                loss_decreasing = metrics.get('loss_decreasing', True)

                qian_strength = min(1.0, accuracy)  # Heaven - creative potential
                dui_strength = min(1.0, 0.5 + (0.5 if loss_decreasing else 0.0))  # Lake - joy

                trigram_strengths['qian'] += qian_strength
                trigram_strengths['dui'] += dui_strength

        # Normalize strengths
        max_strength = max(trigram_strengths.values())
        if max_strength > 0:
            for trigram in trigram_strengths:
                trigram_strengths[trigram] /= max_strength

        return trigram_strengths

    def is_auspicious_timing(self, intervention_type):
        """Determine if timing is auspicious for a specific intervention type"""
        # Different interventions align with different cosmic cycles
        if intervention_type == 'activate':
            # Activation is best at beginning of cycles (like dawn)
            return (self.step_counter % self.timing_cycles['short']) < 2

        elif intervention_type == 'stabilize':
            # Stabilization is best at middle of cycles (like zenith)
            return abs((self.step_counter % self.timing_cycles['medium']) -
                       (self.timing_cycles['medium'] // 2)) < 3

        elif intervention_type == 'balance':
            # Balance is best at transition points (like equinox)
            return (self.step_counter % self.timing_cycles['medium']) in [7, 14, 21]

        elif intervention_type == 'transform':
            # Transformation is best at endings (like sunset)
            return (self.step_counter % self.timing_cycles['short']) >= (self.timing_cycles['short'] - 2)

        elif intervention_type == 'harmonize':
            # Harmonization is best at completion points (like solstice)
            return (self.step_counter % self.timing_cycles['long']) in [0, 30, 59]

        return True  # Default to true for other types

    def find_mechanism_points(self, model, trackers, metrics=None):
        """Find mechanism points for intervention"""
        mechanism_points = []
        self.step_counter += 1

        # Get current element strengths
        element_strengths = {}
        if 'wu_xing_balance' in trackers:
            wu_xing_tracker = trackers['wu_xing_balance']
            if hasattr(wu_xing_tracker, 'get_element_strengths'):
                element_strengths = wu_xing_tracker.get_element_strengths()

        # Get current hexagram
        current_hexagram = None
        if 'hexagram_state' in trackers:
            hex_tracker = trackers['hexagram_state']
            if hasattr(hex_tracker, 'get_current_hexagram_index'):
                current_hexagram = hex_tracker.get_current_hexagram_index()

        # Analyze trigram energy balance
        trigram_strengths = self.analyze_trigram_balance(model, metrics or {})

        # Examine each module in the model
        for i, (name, module) in enumerate(model.named_modules()):
            if isinstance(module, nn.Linear):
                # For each linear layer, determine potential intervention types

                # Get layer statistics
                if hasattr(module, 'weight'):
                    with torch.no_grad():
                        weight_mean = module.weight.mean().item()
                        weight_std = module.weight.std().item()
                        weight_max = module.weight.abs().max().item()
                        weight_sparsity = (module.weight.abs() < 0.01).float().mean().item()

                        # Determine layer position (input, hidden, output)
                        layer_position = "hidden"
                        if "input" in name or i == 0:
                            layer_position = "input"
                        elif "output" in name or i == len(list(model.modules())) - 1:
                            layer_position = "output"

                        # Calculate intervention scores based on layer stats and cosmic factors

                        # 1. Activation intervention: for dead/dormant neurons
                        activation_score = weight_sparsity * 2.0
                        if layer_position == "input" and Element.WATER in element_strengths:
                            activation_score *= 1.0 + element_strengths[Element.WATER] / 100.0

                        # 2. Stabilization intervention: for unstable weights
                        stability_score = min(1.0, weight_std / (0.1 + abs(weight_mean)))
                        if layer_position == "hidden" and Element.EARTH in element_strengths:
                            stability_score *= 1.0 + element_strengths[Element.EARTH] / 100.0

                        # 3. Balance intervention: for imbalanced weights
                        balance_score = min(1.0, abs(weight_mean) / (0.1 + weight_std))
                        if Element.METAL in element_strengths:
                            balance_score *= 1.0 + element_strengths[Element.METAL] / 100.0

                        # 4. Transform intervention: for stale weights
                        transform_score = 0.5  # Base score
                        if layer_position == "hidden" and Element.FIRE in element_strengths:
                            transform_score *= 1.0 + element_strengths[Element.FIRE] / 100.0

                        # 5. Harmonize intervention: for connection flow
                        harmonize_score = min(1.0, weight_max / 5.0)
                        if Element.WOOD in element_strengths:
                            harmonize_score *= 1.0 + element_strengths[Element.WOOD] / 100.0

                        # Adjust scores based on trigram energies
                        if trigram_strengths['kan'] > 0.7:  # Strong Water energy
                            activation_score *= 1.3
                        if trigram_strengths['gen'] > 0.7:  # Strong Mountain energy
                            stability_score *= 1.3
                        if trigram_strengths['kun'] > 0.7:  # Strong Earth energy
                            balance_score *= 1.3
                        if trigram_strengths['li'] > 0.7:  # Strong Fire energy
                            transform_score *= 1.3
                        if trigram_strengths['xun'] > 0.7:  # Strong Wind energy
                            harmonize_score *= 1.3

                        # Select highest scoring intervention type
                        scores = {
                            'activate': activation_score,
                            'stabilize': stability_score,
                            'balance': balance_score,
                            'transform': transform_score,
                            'harmonize': harmonize_score
                        }

                        best_type = max(scores.items(), key=lambda x: x[1])
                        intervention_type, strength = best_type

                        # Check timing aspects
                        if self.is_auspicious_timing(intervention_type):
                            strength *= 1.5  # Boost strength if timing is auspicious

                        # Create intervention description
                        intervention = {
                            'type': intervention_type,
                            'layer': name,
                            'description': f"{intervention_type.capitalize()} the {name} layer",
                            'timing': self.is_auspicious_timing(intervention_type),
                            'hexagram': current_hexagram
                        }

                        # Add to potential mechanism points
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
            success = self.intervention_types[int_type](layer, intervention)

            # Record successful intervention
            if success:
                self.intervention_history.append({
                    'step': self.step_counter,
                    'type': int_type,
                    'layer': layer_name,
                    'timing': intervention.get('timing', False),
                    'hexagram': intervention.get('hexagram', None)
                })

            return success

        return False

    def _activate_intervention(self, layer, intervention):
        """Activation intervention - revitalize dead neurons"""
        if hasattr(layer, 'weight') and layer.weight is not None:
            with torch.no_grad():
                # Find near-zero weights
                mask = (layer.weight.abs() < 0.01)

                # Count how many weights to activate
                total_weights = mask.sum().item()
                if total_weights > 0:
                    # Create Lo Shu pattern for activation
                    lo_shu = LoShuTensor()

                    # Apply small structured noise based on Lo Shu pattern
                    if mask.dim() > 1:
                        # For 2D weight matrices
                        rows, cols = mask.shape
                        for i in range(0, rows, 3):
                            for j in range(0, cols, 3):
                                # Get bounds respecting matrix dimensions
                                i_end = min(i + 3, rows)
                                j_end = min(j + 3, cols)

                                # Get the Lo Shu template for this block
                                template = lo_shu.normalized[:i_end - i, :j_end - j]

                                # Apply to masked weights in this block
                                local_mask = mask[i:i_end, j:j_end]
                                if local_mask.sum() > 0:
                                    layer.weight.data[i:i_end, j:j_end][local_mask] += \
                                        (torch.rand_like(template.view(-1)) * 0.1) * \
                                        template.view(-1)[:local_mask.numel()]
                    else:
                        # For 1D weight vectors, use a flattened Lo Shu pattern
                        template = lo_shu.normalized.view(-1)
                        pattern = template.repeat(math.ceil(mask.numel() / template.numel()))
                        pattern = pattern[:mask.numel()]
                        layer.weight.data[mask] += torch.rand_like(pattern) * 0.1 * pattern

                    return True
        return False

    def _stabilize_intervention(self, layer, intervention):
        """Stabilization intervention - reduce outlier weights"""
        if hasattr(layer, 'weight') and layer.weight is not None:
            with torch.no_grad():
                # Calculate weight statistics
                weight_mean = layer.weight.mean()
                weight_std = layer.weight.std()

                # Find outlier weights
                mask = (layer.weight - weight_mean).abs() > 2 * weight_std

                # Count how many weights to stabilize
                total_weights = mask.sum().item()
                if total_weights > 0:
                    # Move outliers toward the mean with Wu Xing Earth element principles
                    earth_factor = 0.7  # Earth element promotes stability
                    layer.weight.data[mask] = layer.weight.data[mask] * earth_factor + weight_mean * (1 - earth_factor)
                    return True
        return False

    def _balance_intervention(self, layer, intervention):
        """Balance intervention - equalize positive and negative weights"""
        if hasattr(layer, 'weight') and layer.weight is not None:
            with torch.no_grad():
                # Balance positive and negative weights
                pos_mask = layer.weight > 0
                neg_mask = layer.weight < 0

                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    pos_sum = layer.weight[pos_mask].sum()
                    neg_sum = layer.weight[neg_mask].sum()

                    if abs(pos_sum + neg_sum) > 0.1:
                        # Scale to balance with Metal element principles
                        metal_factor = 0.8  # Metal element promotes precision

                        if neg_sum.abs() > 0 and pos_sum.abs() > 0:
                            ratio = pos_sum.abs() / neg_sum.abs()
                            # Apply gentle balancing, not forcing exact equality
                            if ratio > 1.5:
                                layer.weight.data[neg_mask] *= (1 + (ratio - 1) * metal_factor * 0.5)
                            elif ratio < 0.67:
                                layer.weight.data[pos_mask] *= (1 + (1 / ratio - 1) * metal_factor * 0.5)
                            return True
        return False

    def _transform_intervention(self, layer, intervention):
        """Transform intervention - break stagnation patterns"""
        if hasattr(layer, 'weight') and layer.weight is not None:
            with torch.no_grad():
                # Identify stagnant regions (weights that have similar magnitude)
                flat_weights = layer.weight.view(-1)
                sorted_weights, _ = torch.sort(flat_weights.abs())

                # Look for plateaus in the sorted weights
                diffs = sorted_weights[1:] - sorted_weights[:-1]
                small_diffs = (diffs < 0.001).float()

                # If we have a significant plateau, apply transformation
                if small_diffs.mean() > 0.3:  # If 30% of adjacent weights are very similar
                    # Apply Fire element transformation principles
                    fire_factor = 0.2  # Fire element promotes transformation

                    # Create a mask for weights to transform
                    stagnant_mask = torch.zeros_like(flat_weights, dtype=torch.bool)

                    # Identify plateaus of similar weights
                    for i in range(len(small_diffs) - 4):
                        if small_diffs[i:i + 5].sum() >= 4:  # At least 4 adjacent similar weights
                            # Mark the middle of this plateau for transformation
                            stagnant_mask[i + 2:i + 4] = True

                    # Reshape mask back to weight dimensions
                    stagnant_mask = stagnant_mask.view(layer.weight.shape)

                    if stagnant_mask.sum() > 0:
                        # Apply subtle transformation to stagnant weights
                        layer.weight.data[stagnant_mask] *= (
                                    1 + (torch.rand_like(layer.weight[stagnant_mask]) - 0.5) * fire_factor)
                        return True
        return False

    def _harmonize_intervention(self, layer, intervention):
        """Harmonize intervention - improve connection flow"""
        if hasattr(layer, 'weight') and layer.weight is not None:
            with torch.no_grad():
                # Calculate flow patterns between neurons
                if layer.weight.dim() > 1:
                    # For each output neuron, calculate how balanced its inputs are
                    input_strengths = layer.weight.abs().mean(dim=1, keepdim=True)
                    normalized_weights = layer.weight / (input_strengths + 1e-6)

                    # Find neurons with unbalanced input connections
                    imbalance = normalized_weights.var(dim=1)
                    unbalanced_mask = imbalance > imbalance.mean() + imbalance.std()

                    # Apply Wood element harmonization principles
                    wood_factor = 0.15  # Wood element promotes growth and flow

                    if unbalanced_mask.sum() > 0:
                        for i, is_unbalanced in enumerate(unbalanced_mask):
                            if is_unbalanced:
                                # Get this neuron's weights
                                weights = layer.weight[i]

                                # Calculate mean and standard deviation
                                w_mean = weights.mean()
                                w_std = weights.std()

                                # Move extreme weights slightly toward mean
                                extreme_mask = (weights - w_mean).abs() > 1.5 * w_std
                                if extreme_mask.sum() > 0:
                                    # Apply gentle harmonization
                                    layer.weight.data[i][extreme_mask] = layer.weight.data[i][extreme_mask] * (
                                                1 - wood_factor) + \
                                                                         w_mean * wood_factor

                        return True
        return False


#################################################
# Celestial Timing Scheduler
#################################################

class CelestialTimingScheduler:
    """
    Learning rate scheduler based on celestial timing principles.
    Aligns training phases with optimal cosmic cycles.
    """

    def __init__(self, optimizer, hexagram_tracker, wu_xing_tracker, base_lr=0.01, min_lr=1e-6, max_lr=1e-2):
        """
        Initialize CelestialTimingScheduler with improved strategy.
        """
        # Validate optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")

        self.optimizer = optimizer
        self.hexagram_tracker = hexagram_tracker
        self.wu_xing_tracker = wu_xing_tracker
        self.bagua = Bagua()
        self.min_lr = min_lr
        self.max_lr = max_lr

        # Initialize state tracking
        self.step_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Define phases aligned with Later Heaven sequence
        self.training_phases = {
            'initialization': self.bagua.later_heaven_sequence[0],  # Kan (Water)
            'early_training': self.bagua.later_heaven_sequence[1],  # Kun (Earth)
            'acceleration': self.bagua.later_heaven_sequence[2],  # Zhen (Thunder)
            'stabilization': self.bagua.later_heaven_sequence[3],  # Gen (Mountain)
            'refinement': self.bagua.later_heaven_sequence[4],  # Qian (Heaven)
            'completion': self.bagua.later_heaven_sequence[5],  # Dui (Lake)
        }

        # Current phase
        self.current_phase = 'initialization'
        self.phase_step_count = 0

        # Cosmic cycles for timing
        self.cycles = {
            'diurnal': 12,  # 12-step day/night cycle
            'lunar': 28,  # 28-step lunar month
            'seasonal': 72,  # 72-step seasonal cycle
            'annual': 360  # 360-step annual cycle
        }

        # Element factors for different phases
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

        # Calculate cosmic cycle positions
        diurnal_position = self.step_count % self.cycles['diurnal']
        lunar_position = self.step_count % self.cycles['lunar']
        seasonal_position = self.step_count % self.cycles['seasonal']
        annual_position = self.step_count % self.cycles['annual']

        # Determine phase based on training progress
        self._update_training_phase(metrics)

        # Get current trigram for this phase
        current_trigram = self.training_phases[self.current_phase]

        # Get current element
        current_element = None
        if hasattr(self.wu_xing_tracker, 'get_dominant_element'):
            current_element = self.wu_xing_tracker.get_dominant_element()

        # Calculate cosmic timing factor
        timing_factor = self._calculate_timing_factor(
            diurnal_position, lunar_position, seasonal_position, annual_position,
            current_trigram, current_element
        )

        # Apply element factor if available
        element_factor = 1.0
        if current_element is not None:
            element_factor = self.element_factors.get(current_element, 1.0)

        # Combined factor with learning rate decay
        base_decay = 0.995 ** (self.step_count / 100)
        combined_factor = timing_factor * element_factor * base_decay

        # Update learning rates with bounds
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = max(self.min_lr, min(self.max_lr, self.base_lrs[i] * combined_factor))

        self.phase_step_count += 1

        return [group['lr'] for group in self.optimizer.param_groups]

    def _update_training_phase(self, metrics):
        """Update the training phase based on metrics and progress"""
        if metrics is None:
            metrics = {}

        # Get important metrics
        loss = metrics.get('loss', 1.0)
        accuracy = metrics.get('accuracy', 0.0)
        epoch = metrics.get('epoch', self.step_count // 100)

        # Determine appropriate phase based on training progress
        new_phase = self.current_phase

        if epoch < 5:
            new_phase = 'initialization'
        elif epoch < 15:
            new_phase = 'early_training'
        elif loss > 0.3:
            new_phase = 'acceleration'
        elif accuracy < 0.9:
            new_phase = 'stabilization'
        elif accuracy >= 0.9 and accuracy < 0.95:
            new_phase = 'refinement'
        else:
            new_phase = 'completion'

        # Phase transition logic
        if new_phase != self.current_phase:
            self._apply_phase_transition(new_phase)
            self.current_phase = new_phase
            self.phase_step_count = 0

    def _calculate_timing_factor(self, diurnal, lunar, seasonal, annual, trigram, element):
        """Calculate timing factor based on cosmic cycles"""
        # Base timing factor
        factor = 1.0

        # Diurnal cycle effect (like hours of the day)
        if diurnal < 3:  # Dawn - increasing energy
            factor *= 1.1
        elif diurnal < 6:  # Morning - high energy
            factor *= 1.2
        elif diurnal < 9:  # Afternoon - stable energy
            factor *= 1.0
        else:  # Evening - decreasing energy
            factor *= 0.9

        # Lunar cycle effect
        moon_phase = lunar / self.cycles['lunar']
        # New moon (0) and full moon (0.5) are high energy points
        lunar_factor = 1.0 + 0.2 * math.cos(2 * math.pi * moon_phase)
        factor *= lunar_factor

        # Seasonal effect
        season = (seasonal / self.cycles['seasonal']) * 4
        if season < 1:  # Spring - growth
            seasonal_factor = 1.2
        elif season < 2:  # Summer - peak
            seasonal_factor = 1.5
        elif season < 3:  # Autumn - harvest
            seasonal_factor = 0.9
        else:  # Winter - rest
            seasonal_factor = 0.7
        factor *= seasonal_factor

        # Trigram-specific adjustments
        if trigram == 'kan':  # Water - flow
            factor *= 1.0
        elif trigram == 'li':  # Fire - transformation
            factor *= 1.2
        elif trigram == 'zhen':  # Thunder - activation
            factor *= 1.3
        elif trigram == 'gen':  # Mountain - stability
            factor *= 0.8
        elif trigram == 'kun':  # Earth - receptivity
            factor *= 0.9
        elif trigram == 'qian':  # Heaven - creativity
            factor *= 1.1

        return factor if factor > 0.0001 else 0.0001

    def _apply_phase_transition(self, new_phase):
        """Apply more gradual adjustments during phase transitions"""
        if new_phase == 'initialization':
            # Higher learning rates for exploration
            for i, group in enumerate(self.optimizer.param_groups):
                self.base_lrs[i] = self.max_lr

        elif new_phase == 'acceleration':
            # More moderate boost - reduced from 1.5x to 1.2x
            for i, group in enumerate(self.optimizer.param_groups):
                self.base_lrs[i] = min(self.max_lr, self.base_lrs[i] * 1.2)

        elif new_phase == 'stabilization':
            # Much gentler reduction - increased from 0.7x to 0.9x
            for i, group in enumerate(self.optimizer.param_groups):
                self.base_lrs[i] = max(self.min_lr * 10, self.base_lrs[i] * 0.9)

        elif new_phase == 'refinement':
            # Gentler reduction - increased from 0.5x to 0.7x
            for i, group in enumerate(self.optimizer.param_groups):
                self.base_lrs[i] = max(self.min_lr * 10, self.base_lrs[i] * 0.7)

        elif new_phase == 'completion':
            # Less dramatic minimum - increased from 0.3x to 0.5x
            for i, group in enumerate(self.optimizer.param_groups):
                self.base_lrs[i] = max(self.min_lr * 10, self.base_lrs[i] * 0.5)

    def update_with_metrics(self, loss, accuracy, trackers):
        """Update learning rate based on metrics and trackers"""
        metrics = {
            'loss': loss,
            'accuracy': accuracy
        }

        # Apply standard step
        return self.step(metrics)

#################################################
# DualLayer Mechanism Optimizer
#################################################

class DualLayerMechanismOptimizer:
    """
    A meta-optimizer that combines regular optimization with mechanism-based interventions.

    This approach divides the optimization process into two complementary layers:
    1. Regular, continuous updates using an inner optimizer (Wu Wei)
    2. Strategic, targeted interventions at mechanism points using CelestialMechanismGrabber
    """

    def __init__(self, model, base_optimizer, mechanism_grabber=None,
                 intervention_frequency=10, mechanism_threshold=1.5,
                 scheduler=None):
        """
        Initialize the dual-layer optimizer.
        """
        # Validate inputs
        if not hasattr(model, 'parameters'):
            raise TypeError("Model must have parameters() method")

        if not isinstance(base_optimizer, torch.optim.Optimizer):
            raise TypeError("base_optimizer must be a torch.optim.Optimizer instance")

        self.model = model
        self.base_optimizer = base_optimizer
        self.mechanism_grabber = mechanism_grabber or CelestialMechanismGrabber()
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
        """Clear gradients of all parameters"""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, loss=None, accuracy=None, closure=None):
        """Perform a single optimization step with dual-layer approach"""
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
                self.model, self.trackers,
                {'loss': loss, 'accuracy': accuracy, 'loss_decreasing': loss_decreasing}
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
            if hasattr(hex_tracker, 'get_current_hexagram_index') and hasattr(hex_tracker,
                                                                              'get_current_hexagram_name'):
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

#################################################
# Lo Shu Magic Square Neural Network
#################################################

class LoShuLayer(nn.Module):
    """
    Neural network layer based on the Lo Shu magic square.
    Implements a linear transformation with weight initialization and structure
    inspired by the 3x3 Lo Shu square.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(LoShuLayer, self).__init__()

        # Create Lo Shu template
        self.lo_shu = LoShuTensor()

        # Create actual weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize with Lo Shu pattern
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Lo Shu pattern"""
        with torch.no_grad():
            # Get enhanced Lo Shu template
            template = self.lo_shu.get_weight_template(self.weight.size(1), self.weight.size(0))

            # Initialize with scaled template
            scale = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
            self.weight.data = template * torch.randn_like(self.weight) * scale

            if self.bias is not None:
                # Initialize bias with small positive values (Lo Shu always sums to 15)
                fan_in, _ = self._calculate_fan_in_and_fan_out()
                bound = 1 / math.sqrt(fan_in)
                self.bias.data.uniform_(0, bound)

    def _calculate_fan_in_and_fan_out(self):
        """Calculate fan-in and fan-out for proper initialization scaling"""
        fan_in = self.weight.size(1)
        fan_out = self.weight.size(0)
        return fan_in, fan_out

    def forward(self, input):
        """Forward pass with Taiji balance point"""
        # Compute standard linear transformation
        output = F.linear(input, self.weight, self.bias)

        # Apply subtle balancing based on Taiji principle (center of Lo Shu is 5)
        mean_activation = output.mean(dim=1, keepdim=True)
        balanced_output = output + (5.0 / 45.0 - 0.1) * mean_activation

        return balanced_output

class HexagramActivation(nn.Module):
    """
    Neural network activation function based on the 64 hexagrams.
    Dynamically selects the most appropriate activation pattern based on input characteristics.
    """

    def __init__(self):
        super(HexagramActivation, self).__init__()

        # Initialize the eight trigram activations
        self.bagua_activations = BaguaActivations()
        self.hexagrams = Hexagrams()

        # Cache for efficient computation
        self.activation_cache = {}

    def forward(self, x):
        """Apply hexagram-based activation function"""
        # Analyze input tensor characteristics
        batch_size = x.size(0) if x.dim() > 1 else 1

        # Calculate meaningful statistics about the input tensor
        mean_val = x.mean().item()
        std_val = x.std().item()
        max_val = x.max().item()
        min_val = x.min().item()
        sparsity = (x.abs() < 0.01).float().mean().item()

        # Convert these statistics into 6 lines to form a hexagram
        # Each statistic determines a line (yang or yin) based on thresholds
        lines = [
            1 if mean_val > 0 else 0,  # Line 1: mean value
            1 if std_val > 0.5 else 0,  # Line 2: standard deviation
            1 if (max_val - min_val) > 1.0 else 0,  # Line 3: range
            1 if sparsity < 0.5 else 0,  # Line 4: sparsity
            1 if abs(mean_val) < std_val else 0,  # Line 5: relation of mean to std
            1 if torch.sum(x > 0).item() > x.numel() / 2 else 0  # Line 6: positive ratio
        ]

        # Convert lines to hexagram index (0-63)
        hex_index = sum(line * (2 ** i) for i, line in enumerate(reversed(lines)))

        # Check if this hexagram activation is in cache
        if hex_index in self.activation_cache:
            return self.activation_cache[hex_index](x)

        # Get upper and lower trigrams
        upper_trigram = lines[:3]
        lower_trigram = lines[3:]

        # Convert trigrams to names
        upper_key = ''.join(map(str, upper_trigram))
        lower_key = ''.join(map(str, lower_trigram))

        upper_name = self.bagua_activations.bagua.trigram_names.get(upper_key, 'qian')
        lower_name = self.bagua_activations.bagua.trigram_names.get(lower_key, 'qian')

        # Create a composite activation based on the hexagram
        def composite_activation(input_tensor):
            # Apply upper trigram activation to positive values
            positive_mask = input_tensor >= 0
            negative_mask = input_tensor < 0

            result = input_tensor.clone()

            # Apply appropriate activations based on trigrams
            if positive_mask.any():
                result[positive_mask] = self.bagua_activations.forward(
                    input_tensor[positive_mask], upper_name
                )

            if negative_mask.any():
                # Apply lower trigram activation to negative values, but first flip sign
                neg_values = -input_tensor[negative_mask]
                activated_neg = self.bagua_activations.forward(neg_values, lower_name)
                result[negative_mask] = -activated_neg

            return result

        # Cache this activation function
        self.activation_cache[hex_index] = composite_activation

        # Apply the composite activation
        return composite_activation(x)

class LoShuNN(nn.Module):
    """
    Neural network architecture based on the Lo Shu magic square and I-Ching principles.
    Features Lo Shu-structured layers, hexagram activations, and balanced architecture.
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        super(LoShuNN, self).__init__()

        # Input layer
        self.input_layer = LoShuLayer(input_size, hidden_sizes[0])

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(LoShuLayer(hidden_sizes[i], hidden_sizes[i + 1]))

        # Output layer
        self.output_layer = LoShuLayer(hidden_sizes[-1], output_size)

        # Hexagram activation
        self.activation = HexagramActivation()

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = self.activation(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        # Output layer
        x = self.output_layer(x)

        return x

#################################################
# Training and Evaluation Functions
#################################################

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

        # Calculate metrics for trackers
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

    # Plot element strengths
    if 'element_states' in metrics and metrics['element_states']:
        # Extract element strengths over time
        element_data = {e: [] for e in Element}
        for state in metrics['element_states']:
            for e in Element:
                if e in state:
                    element_data[e].append(state[e])
                else:
                    element_data[e].append(0)

        # Plot element strengths
        for element, values in element_data.items():
            if values:  # Check if we have data
                axs[1, 1].plot(values, label=str(element), color=element.color)

        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Element Strength')
        axs[1, 1].set_title('Wu Xing Element Balance')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def compare_models(lo_shu_model, standard_model, X_test, y_test, noise_levels=[0.0, 0.1, 0.2, 0.5]):
    """Compare Lo Shu model with standard model under different noise conditions"""
    results = []

    for noise_level in noise_levels:
        # Add noise to test data
        if noise_level > 0:
            noisy_X = X_test + torch.randn_like(X_test) * noise_level
        else:
            noisy_X = X_test.clone()

        # Evaluate Lo Shu model
        lo_shu_model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = lo_shu_model(noisy_X)
            lo_shu_time = time.time() - start_time

            lo_shu_pred = (torch.sigmoid(outputs) >= 0.5).float()
            lo_shu_acc = (lo_shu_pred == y_test).float().mean().item()

        # Evaluate standard model
        standard_model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = standard_model(noisy_X)
            std_time = time.time() - start_time

            std_pred = (torch.sigmoid(outputs) >= 0.5).float()
            std_acc = (std_pred == y_test).float().mean().item()

        # Record results
        results.append({
            'noise_level': noise_level,
            'lo_shu_accuracy': lo_shu_acc,
            'standard_accuracy': std_acc,
            'lo_shu_time': lo_shu_time,
            'standard_time': std_time,
            'speedup': std_time / lo_shu_time if lo_shu_time > 0 else 0
        })

    # Print results
    print("\nModel Comparison Results:")
    print("-------------------------")
    print(f"{'Noise Level':<12} {'Lo Shu Acc':<12} {'Standard Acc':<12} {'Speedup':<12}")
    print("-" * 50)

    for result in results:
        print(f"{result['noise_level']:<12.2f} {result['lo_shu_accuracy']:<12.4f} "
              f"{result['standard_accuracy']:<12.4f} {result['speedup']:<12.2f}x")

    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot accuracy comparison
    plt.subplot(1, 2, 1)
    plt.plot([r['noise_level'] for r in results],
             [r['lo_shu_accuracy'] for r in results],
             'o-', label='Lo Shu NN')
    plt.plot([r['noise_level'] for r in results],
             [r['standard_accuracy'] for r in results],
             's-', label='Standard NN')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy')
    plt.title('Robustness to Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot speedup
    plt.subplot(1, 2, 2)
    plt.bar([r['noise_level'] for r in results],
            [r['speedup'] for r in results])
    plt.xlabel('Noise Level')
    plt.ylabel('Speedup Factor')
    plt.title('Lo Shu NN Speed Advantage')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results

#################################################
# Main Function: Demonstration
#################################################

def main():
    """Main function demonstrating Daoist Neural Network approach"""
    print("=== Daoist Neural Network Framework ===")
    print("\nInitializing with celestial patterns...")

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X_train, y_train, X_test, y_test = generate_data(n_samples=3000, n_features=9)
    print(f"Data shapes: X_train {X_train.shape}, y_train {y_train.shape}, "
          f"X_test {X_test.shape}, y_test {y_test.shape}")

    # Create Lo Shu model
    print("\nCreating Lo Shu Neural Network...")
    lo_shu_model = LoShuNN(input_size=9, hidden_sizes=[64, 32, 16], output_size=1)

    # Create standard model for comparison
    print("Creating Standard Neural Network for comparison...")
    standard_model = nn.Sequential(
        nn.Linear(9, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

    # Initialize trackers
    print("\nInitializing celestial trackers...")
    trackers = {
        'hexagram_state': HexagramStateTracker(),
        'wu_xing_balance': WuXingSystem()
    }

    # Initialize mechanism grabber
    print("\nInitializing celestial mechanism grabber...")
    mechanism_grabber = CelestialMechanismGrabber()

    # Set up Lo Shu model optimization
    print("\nSetting up optimization approach...")
    base_optimizer = WuWeiOptimizer(
        lo_shu_model.parameters(),
        lr=0.01,
        threshold=0.01,
        wu_wei_ratio=0.7
    )

    # Create scheduler
    lr_scheduler = CelestialTimingScheduler(
        base_optimizer,
        hexagram_tracker=trackers['hexagram_state'],
        wu_xing_tracker=trackers['wu_xing_balance']
    )

    # Create the dual-layer wrapper
    optimizer = DualLayerMechanismOptimizer(
        model=lo_shu_model,
        base_optimizer=base_optimizer,
        mechanism_grabber=mechanism_grabber,
        intervention_frequency=10,
        mechanism_threshold=1.5,
        scheduler=lr_scheduler
    )
    optimizer.register_trackers(trackers)

    # Set up standard model with Adam optimizer
    std_optimizer = torch.optim.Adam(standard_model.parameters(), lr=0.01)

    # Training settings
    epochs = 50
    batch_size = 32

    # Initialize metrics
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'learning_rates': [],
        'element_states': []
    }

    # Train Lo Shu model
    print("\nTraining Lo Shu Neural Network...")
    for epoch in range(epochs):
        # Train one epoch
        train_loss, train_acc = train_epoch(
            lo_shu_model, optimizer, X_train, y_train,
            batch_size=batch_size, trackers=trackers
        )

        # Validate
        val_loss, val_acc = validate(lo_shu_model, X_test, y_test)

        # Record metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_accuracy'].append(train_acc)
        metrics['val_accuracy'].append(val_acc)

        # Record learning rate
        lr = optimizer.param_groups[0]['lr']
        metrics['learning_rates'].append(lr)

        # Record element states
        if 'wu_xing_balance' in trackers:
            element_strengths = trackers['wu_xing_balance'].get_element_strengths()
            metrics['element_states'].append(element_strengths)

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

    print("\nTraining complete for Lo Shu model.")

    # Train standard model
    print("\nTraining Standard Neural Network for comparison...")
    std_metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    for epoch in range(epochs):
        # Train standard model
        standard_model.train()
        train_loss = 0
        correct = 0
        total = 0

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch_X, batch_y in dataloader:
            # Zero gradients
            std_optimizer.zero_grad()

            # Forward pass
            outputs = standard_model(batch_X)

            # Calculate loss
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            std_optimizer.step()

            # Update metrics
            train_loss += loss.item() * batch_X.size(0)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss /= total
        train_acc = correct / total

        # Validate
        val_loss, val_acc = validate(standard_model, X_test, y_test)

        # Record metrics
        std_metrics['train_loss'].append(train_loss)
        std_metrics['val_loss'].append(val_loss)
        std_metrics['train_accuracy'].append(train_acc)
        std_metrics['val_accuracy'].append(val_acc)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    print("\nTraining complete for standard model.")

    # Compare models
    print("\nComparing models...")
    comparison_results = compare_models(lo_shu_model, standard_model, X_test, y_test)

    # Visualize Lo Shu model training results
    print("\nVisualizing Lo Shu model training results...")
    visualize_training(metrics)

    # Final evaluation
    print("\nFinal evaluation:")
    final_lo_shu_loss, final_lo_shu_acc = validate(lo_shu_model, X_test, y_test)
    final_std_loss, final_std_acc = validate(standard_model, X_test, y_test)

    print(f"Lo Shu model - validation loss: {final_lo_shu_loss:.4f}, accuracy: {final_lo_shu_acc:.4f}")
    print(f"Standard model - validation loss: {final_std_loss:.4f}, accuracy: {final_std_acc:.4f}")

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

    print("\n=== Daoist Neural Network Framework Demonstration Complete ===")

if __name__ == "__main__":
    main()