"""
DaoML: A Machine Learning Framework Inspired by Daoist Principles
Combines Lo Shu magic square, Wu Wei optimization, I-Ching state tracking,
Wu Xing elemental cycles, and Taiji balance principles.

Fixed version to resolve the backward graph issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from enum import Enum
import math
import time
from typing import Dict, List, Tuple, Optional, Union

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
        # Extract network metrics - ensure they are all primitive types, not tensors
        loss = float(network_metrics['loss'])
        accuracy = float(network_metrics['accuracy'])
        gradient_norm = float(network_metrics['gradient_norm'])
        activation_mean = float(network_metrics['activation_mean'])
        activation_std = float(network_metrics['activation_std'])
        weight_norm = float(network_metrics['weight_norm'])
        loss_decreasing = bool(network_metrics['loss_decreasing'])

        # Calculate six lines based on network metrics
        lines = [
            # Line 1 (bottom): Loss trend (yang if decreasing, yin if increasing)
            1 if loss_decreasing else 0,

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
    """
    Complete Wu Xing system with all cycles:
    - Generation (生, sheng)
    - Conquest (克, ke)
    - Insult (侮, wu)
    - Mother-Child (母子, mu-zi)
    - Control (乘, cheng)
    - Rebellion (侮乘, wu-cheng)
    - Over-Acting (太過, tai-guo)
    - Under-Acting (不及, bu-ji)
    - Balance (平衡, ping-heng)
    """

    def __init__(self, initial_state: Dict[Element, float] = None):
        """
        Initialize the system with element values.

        Args:
            initial_state: Initial values for each element (0-100)
        """
        # Default initial state: balanced
        if initial_state is None:
            initial_state = {element: 20.0 for element in Element}

        self.state = initial_state.copy()
        self.history = [self.state.copy()]
        self.cycle_effects = {}  # Track effects of each cycle

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

        # Define elemental constraints based on natures
        self.constraints = {
            # Wood cannot directly rise to Fire (seeks rise but cannot)
            (Element.WOOD, Element.FIRE): 0.3,
            # Metal cannot directly fall to Water (seeks descent but cannot)
            (Element.METAL, Element.WATER): 0.3
        }

        # Coefficients for various cycles
        self.coefficients = {
            'generation': 0.3,  # Generation effect strength
            'conquest': 0.4,  # Conquest effect strength
            'insult': 0.3,  # Insult effect strength
            'mother_child': 0.2,  # Mother-child effect strength
            'control': 0.4,  # Control effect strength
            'rebellion': 0.3,  # Rebellion effect strength
            'over_acting': 0.3,  # Over-acting effect strength
            'under_acting': 0.3,  # Under-acting effect strength
            'balance': 0.15  # Balance effect strength
        }

        # Thresholds for various cycles
        self.thresholds = {
            'insult': 2.5,  # Threshold ratio for insult cycle
            'control': 1.8,  # Dominance threshold for control cycle
            'rebellion': 1.5,  # Weakening threshold for rebellion cycle
            'over_acting': 2.0,  # Threshold for over-acting cycle
            'under_acting': 0.4,  # Threshold for under-acting cycle
            'balance': 0.25  # Imbalance threshold for balance cycle (as fraction of average)
        }

    def generation_effect(self, source: Element, target: Element) -> float:
        """Calculate generation effect (source generates target)"""
        # Check if this is a generation relationship
        if self.generation_cycle[source] == target:
            # Apply constraint if it exists
            constraint = self.constraints.get((source, target), 1.0)
            effect = self.coefficients['generation'] * self.state[source] * constraint
            return effect
        return 0.0

    def conquest_effect(self, source: Element, target: Element) -> float:
        """Calculate conquest effect (source conquers target)"""
        # Check if this is a conquest relationship
        if self.conquest_cycle[source] == target:
            # Apply constraint if it exists
            constraint = self.constraints.get((source, target), 1.0)
            effect = -self.coefficients['conquest'] * self.state[source] * constraint
            return effect
        return 0.0

    def insult_effect(self, source: Element, target: Element) -> float:
        """
        Calculate insult effect (source insults target).
        Occurs when source is much stronger than target, and target normally conquers source.
        """
        # Find what conquers source
        for element, conquered in self.conquest_cycle.items():
            if conquered == source and element == target:
                # If source is much stronger than target
                if self.state[source] > self.state[target] * self.thresholds['insult']:
                    effect = -self.coefficients['insult'] * self.state[source]
                    return effect
        return 0.0

    def mother_child_effect(self, source: Element, target: Element) -> float:
        """
        Calculate mother-child effect.
        - If source generates target (source is mother), target gets positive effect
        - If target generates source (target is mother), target gets negative effect (drain)
        """
        # Source is mother of target
        if self.generation_cycle[source] == target:
            effect = self.coefficients['mother_child'] * self.state[source] * 0.5
            return effect

        # Target is mother of source
        if self.generation_cycle[target] == source:
            effect = -self.coefficients['mother_child'] * self.state[source] * 0.3
            return effect

        return 0.0

    def control_effect(self, source: Element, target: Element) -> float:
        """
        Calculate control effect (excessive conquest).
        Occurs when source is much stronger than target and normally conquers it.
        """
        # Check if this is a conquest relationship
        if self.conquest_cycle[source] == target:
            # If source is much stronger than target
            if self.state[source] > self.state[target] * self.thresholds['control']:
                effect = -self.coefficients['control'] * (self.state[source] ** 2) / self.state[target]
                return effect
        return 0.0

    def rebellion_effect(self, source: Element, target: Element) -> float:
        """
        Calculate rebellion effect.
        Occurs when source should be conquered by target, but target is weakened by its conquerer.
        """
        # Find what conquers target
        target_conquerer = None
        for element, conquered in self.conquest_cycle.items():
            if conquered == target:
                target_conquerer = element
                break

        # Check if target normally conquers source
        if self.conquest_cycle.get(target) == source and target_conquerer is not None:
            # If target's conquerer is strong enough to weaken target
            if self.state[target_conquerer] > self.state[target] * self.thresholds['rebellion']:
                effect = self.coefficients['rebellion'] * self.state[source] * self.state[target_conquerer] / \
                         self.state[target]
                return effect
        return 0.0

    def over_acting_effect(self, source: Element, target: Element) -> float:
        """
        Calculate over-acting effect.
        Occurs when source generates target and is much stronger than needed.
        """
        # Check if this is a generation relationship
        if self.generation_cycle[source] == target:
            # If source is much stronger than target
            if self.state[source] > self.state[target] * self.thresholds['over_acting']:
                effect = self.coefficients['over_acting'] * (
                            self.state[source] - self.thresholds['over_acting'] * self.state[target])
                return effect
        return 0.0

    def under_acting_effect(self, source: Element, target: Element) -> float:
        """
        Calculate under-acting effect.
        Occurs when source generates target but is too weak to properly support it.
        """
        # Check if this is a generation relationship
        if self.generation_cycle[source] == target:
            # If source is much weaker than target needs
            if self.state[source] < self.state[target] * self.thresholds['under_acting']:
                effect = -self.coefficients['under_acting'] * (
                            self.thresholds['under_acting'] * self.state[target] - self.state[source])
                return effect
        return 0.0

    def balance_effect(self, source: Element, target: Element) -> float:
        """
        Calculate balance effect.
        Tendency of the system to maintain equilibrium.
        """
        # Calculate average element strength
        avg_strength = sum(self.state.values()) / len(self.state)

        # Check if this is a generation relationship
        if self.generation_cycle[source] == target:
            # If target is significantly different from average
            if abs(self.state[target] - avg_strength) > avg_strength * self.thresholds['balance']:
                # Move toward average
                effect = self.coefficients['balance'] * (avg_strength - self.state[target])
                return effect
        return 0.0

    def step(self, dt: float = 0.1) -> Dict[Element, float]:
        """
        Advance the system by one time step.

        Args:
            dt: Time step size

        Returns:
            New system state
        """
        new_state = self.state.copy()

        # Reset cycle effects
        self.cycle_effects = {
            'generation': {e: 0.0 for e in Element},
            'conquest': {e: 0.0 for e in Element},
            'insult': {e: 0.0 for e in Element},
            'mother_child': {e: 0.0 for e in Element},
            'control': {e: 0.0 for e in Element},
            'rebellion': {e: 0.0 for e in Element},
            'over_acting': {e: 0.0 for e in Element},
            'under_acting': {e: 0.0 for e in Element},
            'balance': {e: 0.0 for e in Element}
        }

        # Calculate all interactions
        for source in Element:
            for target in Element:
                if source == target:
                    continue

                # Calculate effects from all cycles
                gen_effect = self.generation_effect(source, target) * dt
                con_effect = self.conquest_effect(source, target) * dt
                ins_effect = self.insult_effect(source, target) * dt
                mc_effect = self.mother_child_effect(source, target) * dt
                ctrl_effect = self.control_effect(source, target) * dt
                reb_effect = self.rebellion_effect(source, target) * dt
                over_effect = self.over_acting_effect(source, target) * dt
                under_effect = self.under_acting_effect(source, target) * dt
                bal_effect = self.balance_effect(source, target) * dt

                # Update target element with all effects
                new_state[target] += gen_effect
                new_state[target] += con_effect
                new_state[target] += ins_effect
                new_state[target] += mc_effect
                new_state[target] += ctrl_effect
                new_state[target] += reb_effect
                new_state[target] += over_effect
                new_state[target] += under_effect
                new_state[target] += bal_effect

                # Track effects by cycle
                self.cycle_effects['generation'][target] += gen_effect
                self.cycle_effects['conquest'][target] += con_effect
                self.cycle_effects['insult'][target] += ins_effect
                self.cycle_effects['mother_child'][target] += mc_effect
                self.cycle_effects['control'][target] += ctrl_effect
                self.cycle_effects['rebellion'][target] += reb_effect
                self.cycle_effects['over_acting'][target] += over_effect
                self.cycle_effects['under_acting'][target] += under_effect
                self.cycle_effects['balance'][target] += bal_effect

                # Conservation of energy: source element decreases when generating
                if gen_effect > 0:
                    new_state[source] -= gen_effect * 0.5

        # Apply natural tendencies based on elemental nature
        for element in Element:
            if element == Element.FIRE:  # Fire rises
                new_state[element] *= (1 + 0.01 * dt)  # Natural increase
            elif element == Element.WATER:  # Water descends
                new_state[element] *= (1 - 0.01 * dt)  # Natural decrease
            elif element == Element.EARTH:  # Earth stabilizes
                # Move toward the average
                avg = sum(new_state.values()) / len(new_state)
                new_state[element] += (avg - new_state[element]) * 0.1 * dt

        # Ensure values stay in reasonable range
        for element in Element:
            new_state[element] = max(0.1, min(100.0, new_state[element]))

        self.state = new_state
        self.history.append(self.state.copy())

        return self.state

    def get_dominant_element(self) -> Element:
        """Return the currently dominant element"""
        return max(self.state.items(), key=lambda x: x[1])[0]

    def apply_intervention(self, target_element: Element, strength: float = 10.0) -> None:
        """
        Apply an intervention to strengthen a specific element.

        Args:
            target_element: Element to strengthen
            strength: Amount to increase
        """
        self.state[target_element] += strength
        self.history.append(self.state.copy())

    def get_element_strengths(self):
        """
        Get the current strengths of all elements.

        Returns:
            Dictionary mapping elements to their current strength values
        """
        return self.state

    def update_batch(self, model, network_metrics):
        """
        Update the Wu Xing system state based on batch metrics

        Args:
            model: The neural network model
            network_metrics: Dictionary of network metrics
        """
        # Adjust element strengths based on network metrics - ensure metrics are primitive types
        loss_decreasing = bool(network_metrics.get('loss_decreasing', False))
        accuracy = float(network_metrics.get('accuracy', 0.0))
        gradient_norm = float(network_metrics.get('gradient_norm', 0.0))
        weight_norm = float(network_metrics.get('weight_norm', 0.0))

        # Water (adaptability) strengthened by decreasing loss, varied gradients
        if loss_decreasing:
            self.state[Element.WATER] += 0.1
        if gradient_norm > 0.5:
            self.state[Element.WATER] += 0.05

        # Wood (growth) strengthened by increasing accuracy
        if accuracy > 0.7:
            self.state[Element.WOOD] += 0.1

        # Fire (transformation) strengthened by high gradient activity
        if gradient_norm > 1.0:
            self.state[Element.FIRE] += 0.15

        # Earth (stability) strengthened by balanced metrics
        if 0.7 < accuracy < 0.9:
            self.state[Element.EARTH] += 0.1
        if 0.5 < weight_norm < 2.0:
            self.state[Element.EARTH] += 0.05

        # Metal (precision) strengthened by high accuracy, low gradients
        if accuracy > 0.9:
            self.state[Element.METAL] += 0.15
        if gradient_norm < 0.1:
            self.state[Element.METAL] += 0.05

        # Apply system dynamics
        self.step(dt=0.01)

    def update_epoch(self, model, network_metrics):
        """
        Update the Wu Xing system state based on epoch metrics

        Args:
            model: The neural network model
            network_metrics: Dictionary of network metrics
        """
        # Similar to update_batch but with larger effects for epoch-level updates
        # Ensure metrics are primitive types
        loss_decreasing = bool(network_metrics.get('loss_decreasing', False))
        accuracy = float(network_metrics.get('accuracy', 0.0))
        gradient_norm = float(network_metrics.get('gradient_norm', 0.0))
        weight_norm = float(network_metrics.get('weight_norm', 0.0))

        # Water (adaptability) strengthened by decreasing loss, varied gradients
        if loss_decreasing:
            self.state[Element.WATER] += 1.0
        if gradient_norm > 0.5:
            self.state[Element.WATER] += 0.5

        # Wood (growth) strengthened by increasing accuracy
        if accuracy > 0.7:
            self.state[Element.WOOD] += 1.0

        # Fire (transformation) strengthened by high gradient activity
        if gradient_norm > 1.0:
            self.state[Element.FIRE] += 1.5

        # Earth (stability) strengthened by balanced metrics
        if 0.7 < accuracy < 0.9:
            self.state[Element.EARTH] += 1.0
        if 0.5 < weight_norm < 2.0:
            self.state[Element.EARTH] += 0.5

        # Metal (precision) strengthened by high accuracy, low gradients
        if accuracy > 0.9:
            self.state[Element.METAL] += 1.5
        if gradient_norm < 0.1:
            self.state[Element.METAL] += 0.5

        # Apply system dynamics with a larger time step
        self.step(dt=0.1)


#################################################
# Lo Shu Neural Network Implementation
#################################################

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


#################################################
# Enhanced Optimizer and Learning Rate Components
#################################################

class ImprovedWuWeiOptimizer(torch.optim.Optimizer):
    """
    Enhanced Wu Wei Optimizer with dynamic threshold adaptation, strategic non-action,
    and natural flow guidance based on the Daoist principle of Wu Wei (non-action).
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999),
                 threshold=0.01, adapt_rate=0.01, wu_wei_ratio=0.7,
                 use_momentum_guidance=True, harmonic_update=True):
        """
        Initialize the enhanced Wu Wei Optimizer.

        Args:
            params: Model parameters (list or iterator)
            lr: Learning rate
            betas: Coefficients for computing running averages
            threshold: Initial gradient threshold for parameter updates
            adapt_rate: Rate at which thresholds adapt (0-1)
            wu_wei_ratio: Proportion of parameters to leave unchanged (0-1)
            use_momentum_guidance: Whether to use momentum for natural flow guidance
            harmonic_update: Whether to use harmonic (gradual) updates for medium significance
        """
        # Ensure params is a list to avoid empty parameter issues
        if not isinstance(params, (list, tuple)):
            params = list(params)

        # Validate params is not empty
        if len(params) == 0:
            raise ValueError("Optimizer got an empty parameter list")

        defaults = dict(lr=lr, betas=betas, threshold=threshold,
                        adapt_rate=adapt_rate, wu_wei_ratio=wu_wei_ratio,
                        use_momentum_guidance=use_momentum_guidance,
                        harmonic_update=harmonic_update)

        super(ImprovedWuWeiOptimizer, self).__init__(params, defaults)

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

        # Parameter momentum for natural flow guidance
        self.param_momentum = {}

        # Success tracking to optimize wu_wei decisions
        self.success_history = {}

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
            use_momentum_guidance = group['use_momentum_guidance']
            harmonic_update = group['harmonic_update']

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

                # Initialize success tracking if needed
                if param_id not in self.success_history:
                    self.success_history[param_id] = {
                        'high_success': 1.0,  # Initial success rate
                        'med_success': 1.0,
                        'low_success': 1.0,
                        'updates': 0
                    }

                # Calculate gradient significance
                if len(p.shape) > 1:  # For matrices, calculate per-output significance
                    grad_significance = p.grad.abs().mean(dim=1)

                    # Get proposed updates
                    proposed_update = p.data - state['prev_params']
                    update_significance = proposed_update.abs().mean(dim=1)

                    # Apply momentum guidance if enabled
                    if use_momentum_guidance:
                        if param_id not in self.param_momentum:
                            self.param_momentum[param_id] = torch.zeros_like(grad_significance).unsqueeze(1)

                        momentum = self.param_momentum[param_id]

                        # Calculate update alignment with momentum
                        update_dir = proposed_update / (update_significance.unsqueeze(1) + 1e-8)
                        momentum_dir = momentum / (momentum.norm(dim=1, keepdim=True) + 1e-8)

                        # Calculate cosine similarity
                        alignment = torch.sum(update_dir * momentum_dir, dim=1)

                        # Apply adaptive amplification based on alignment
                        alignment_factor = (0.5 + 0.5 * alignment).unsqueeze(1)
                        proposed_update = proposed_update * alignment_factor

                        # Update momentum - detach to prevent graph retention
                        self.param_momentum[param_id] = 0.9 * momentum + 0.1 * proposed_update.detach()

                    # Determine which outputs to update based on thresholds
                    current_threshold = self.param_thresholds[param_id]
                    update_mask = (grad_significance > current_threshold).float().unsqueeze(1)

                    # Apply selective update with three tiers
                    high_mask = (grad_significance > 2 * current_threshold).float().unsqueeze(1)
                    med_mask = ((grad_significance > current_threshold) &
                                (grad_significance <= 2 * current_threshold)).float().unsqueeze(1)

                    # Reset to previous values
                    p.data.copy_(state['prev_params'])

                    # Apply tiered updates with adaptive rates based on success history
                    high_rate = 1.0  # Full update for high significance
                    med_rate = 0.5 if harmonic_update else 1.0  # Partial or full update for medium

                    # Apply success-weighted updates
                    success_ratio = self.success_history[param_id]['high_success'] / max(0.1,
                                                                                         self.success_history[param_id][
                                                                                             'low_success'])
                    adaptive_high_rate = min(1.2, max(0.8, success_ratio)) * high_rate

                    success_ratio = self.success_history[param_id]['med_success'] / max(0.1,
                                                                                        self.success_history[param_id][
                                                                                            'low_success'])
                    adaptive_med_rate = min(1.0, max(0.2, success_ratio)) * med_rate

                    # Apply the updates
                    p.data += high_mask * proposed_update * adaptive_high_rate
                    p.data += med_mask * proposed_update * adaptive_med_rate

                    # Count updated parameters
                    update_count += (update_mask.sum() * p.shape[1]).item()

                else:  # For vectors, use element-wise significance
                    grad_significance = p.grad.abs()

                    # Get proposed updates
                    proposed_update = p.data - state['prev_params']
                    update_significance = proposed_update.abs()

                    # Apply momentum guidance if enabled
                    if use_momentum_guidance:
                        if param_id not in self.param_momentum:
                            self.param_momentum[param_id] = torch.zeros_like(proposed_update)

                        momentum = self.param_momentum[param_id]

                        # Calculate alignment (cosine similarity)
                        update_dir = proposed_update / (update_significance + 1e-8)
                        momentum_dir = momentum / (momentum.norm() + 1e-8)
                        alignment = torch.sum(update_dir * momentum_dir) / (
                                    update_dir.norm() * momentum_dir.norm() + 1e-8)

                        # Apply adaptive amplification
                        if alignment > 0.7:  # Strong alignment
                            proposed_update = proposed_update * (1.0 + 0.1 * alignment)

                        # Update momentum - detach to prevent graph retention
                        self.param_momentum[param_id] = 0.9 * momentum + 0.1 * proposed_update.detach()

                    # Determine which elements to update based on thresholds
                    current_threshold = self.param_thresholds[param_id]
                    update_mask = (grad_significance > current_threshold).float()

                    # Apply selective update with three tiers
                    high_mask = (grad_significance > 2 * current_threshold).float()
                    med_mask = ((grad_significance > current_threshold) &
                                (grad_significance <= 2 * current_threshold)).float()

                    # Reset to previous values
                    p.data.copy_(state['prev_params'])

                    # Apply tiered updates with adaptive rates
                    high_rate = 1.0
                    med_rate = 0.5 if harmonic_update else 1.0

                    # Apply success-weighted updates
                    success_ratio = self.success_history[param_id]['high_success'] / max(0.1,
                                                                                         self.success_history[param_id][
                                                                                             'low_success'])
                    adaptive_high_rate = min(1.2, max(0.8, success_ratio)) * high_rate

                    success_ratio = self.success_history[param_id]['med_success'] / max(0.1,
                                                                                        self.success_history[param_id][
                                                                                            'low_success'])
                    adaptive_med_rate = min(1.0, max(0.2, success_ratio)) * med_rate

                    # Apply the updates
                    p.data += high_mask * proposed_update * adaptive_high_rate
                    p.data += med_mask * proposed_update * adaptive_med_rate

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
                    current_ratio = update_count / max(1, total_params)
                    target_ratio = 1 - wu_wei_ratio

                    if current_ratio > target_ratio:
                        # Too many updates, increase threshold
                        self.param_thresholds[param_id] += adapt_rate * median_grad
                    else:
                        # Too few updates, decrease threshold
                        self.param_thresholds[param_id] = max(0.001,
                                                              self.param_thresholds[
                                                                  param_id] - adapt_rate * median_grad)

                # Store current parameters for next iteration
                state['prev_params'] = p.data.clone().detach()  # Detach to prevent graph retention

                # Record update for this parameter
                self.success_history[param_id]['updates'] += 1

        # Track update ratio for monitoring
        if total_params > 0:
            self.update_ratios.append(update_count / total_params)

            # Keep only recent history
            if len(self.update_ratios) > 100:
                self.update_ratios = self.update_ratios[-100:]

        return loss

    def update_success_rates(self, loss_improvement):
        """Update success rates based on loss improvement"""
        # Positive improvement means the update was successful
        improvement_factor = 1.0 + 0.1 * np.sign(loss_improvement)  # Use numpy to avoid tensors

        for param_id, history in self.success_history.items():
            # Update success rates with exponential moving average
            if history['updates'] > 0:
                # High significance updates
                history['high_success'] = 0.9 * history['high_success'] + 0.1 * improvement_factor
                # Medium significance updates
                history['med_success'] = 0.95 * history['med_success'] + 0.05 * improvement_factor
                # Low significance (no update) - assume neutral
                history['low_success'] = 0.99 * history['low_success'] + 0.01

                # Reset update counter
                history['updates'] = 0

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


class IChingTransitionAdvantage:
    """Optimize training by leveraging favorable hexagram transitions"""

    def __init__(self, hexagram_tracker):
        self.hexagram_tracker = hexagram_tracker
        self.favorable_transitions = self._define_favorable_transitions()
        self.current_adjustment = 1.0  # No adjustment initially
        self.recent_adjustments = []  # Keep track of recent adjustments
        self.max_history = 5  # Limit history length for smoothing

    def _define_favorable_transitions(self):
        """Define transitions that are favorable for optimization"""
        favorable = {
            # From difficult situations to better ones
            (22, 40): 1.2,  # Oppression (Kun) to Deliverance (Jie) - boost learning
            (29, 30): 1.15,  # The Abysmal (Kan) to The Clinging (Li) - water to fire
            (33, 34): 1.15,  # Retreat (Dun) to Great Power (Da Zhuang)
            (12, 11): 1.1,  # Standstill (Pi) to Peace (Tai)
            (61, 34): 1.1,  # Inner Truth to Great Power - boost
            (4, 11): 1.05,  # Youthful Folly to Peace - slight boost

            # Unfavorable transitions - reduce learning
            (11, 12): 0.85,  # Peace to Standstill - slow down
            (53, 23): 0.9,  # Gradual Progress to Splitting Apart
            (54, 38): 0.95,  # The Gentle to Opposition - slight caution
            (30, 29): 0.9,  # The Clinging to The Abysmal - fire to water
            (34, 33): 0.9,  # Great Power to Retreat - slow down

            # Significant transitions
            (62, 22): 0.7,  # Great Taming to Oppression - major slowdown
            (22, 62): 1.3,  # Oppression to Great Taming - major boost

            # Generic transition types
            ("identical", None): 1.0,  # No change
            ("single_line", None): 1.02,  # Single line change - slight boost
            ("opposite", None): 0.8,  # Total reversal - significant reduction
            ("nuclear", None): 1.05  # Nuclear hexagram - boost (core insight)
        }
        return favorable

    def calculate_adjustment(self):
        """Calculate learning rate adjustment based on recent transition"""
        if len(self.hexagram_tracker.hexagram_transition_history) < 1:
            return 1.0

        # Get most recent transition
        from_hex, to_hex = self.hexagram_tracker.hexagram_transition_history[-1]

        # Get transition type
        transition_type = self.hexagram_tracker.hexagrams.get_hexagram_transition_type(from_hex, to_hex)

        # Check for specific transition
        adjustment = self.favorable_transitions.get((from_hex, to_hex), None)

        # If no specific transition found, use generic type
        if adjustment is None:
            adjustment = self.favorable_transitions.get((transition_type, None), 1.0)

        # Add to recent adjustments with smooth transition
        self.recent_adjustments.append(adjustment)
        if len(self.recent_adjustments) > self.max_history:
            self.recent_adjustments.pop(0)

        # Calculate weighted average (more weight to recent)
        weights = [0.5 ** i for i in range(len(self.recent_adjustments))]
        weighted_sum = sum(w * adj for w, adj in zip(weights, reversed(self.recent_adjustments)))
        weight_sum = sum(weights)

        # Update current adjustment
        self.current_adjustment = weighted_sum / weight_sum

        return self.current_adjustment


class WuXingLRHarmony:
    """Adaptive learning rate based on elemental harmony principles"""

    def __init__(self, wu_xing_system):
        self.wu_xing = wu_xing_system
        self.cycle_pattern_strengths = {
            'generation': 0.0,  # Strength of generation cycle
            'conquest': 0.0,  # Strength of conquest cycle
            'balance': 0.0  # Overall balance
        }
        self.history = []

    def _calculate_cycle_strengths(self):
        """Calculate the strengths of different elemental cycles"""
        elements = self.wu_xing.state

        # Generation cycle strength
        gen_strength = 0
        for source, target in self.wu_xing.generation_cycle.items():
            # Strong generation when source is strong and target is growing
            source_val = elements[source]
            target_val = elements[target]
            if source_val > 30 and target_val > 10:
                gen_strength += 0.2 * (source_val / 100) * (target_val / 100)

        # Conquest cycle strength
        con_strength = 0
        for source, target in self.wu_xing.conquest_cycle.items():
            # Strong conquest when source is strong and target is being controlled
            source_val = elements[source]
            target_val = elements[target]
            if source_val > 40 and target_val < 30:
                con_strength += 0.15 * (source_val / 100) * (1 - target_val / 100)

        # Balance calculation
        values = list(elements.values())
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        normalized_variance = variance / (mean_val ** 2)
        balance = max(0, 1 - normalized_variance)

        self.cycle_pattern_strengths = {
            'generation': min(1.0, gen_strength),
            'conquest': min(1.0, con_strength),
            'balance': balance
        }

    def get_harmony_factor(self):
        """Calculate harmony factor based on element balance"""
        self._calculate_cycle_strengths()
        elements = self.wu_xing.state

        # Calculate element balance ratio
        min_element = min(elements.values())
        max_element = max(elements.values())

        if max_element < 1e-6:  # Prevent division by zero
            return 1.0

        balance_ratio = min_element / max_element

        # Apply three principles:
        # 1. Perfect balance (all elements equal) - moderately high learning
        # 2. Extreme imbalance - reduce learning to stabilize
        # 3. Productive imbalance - certain imbalance patterns are beneficial

        # Principle 1 & 2: Balance factor
        if balance_ratio > 0.8:  # Close to perfect balance
            balance_factor = 1.1  # Slightly increased learning
        elif balance_ratio < 0.2:  # Extreme imbalance
            balance_factor = 0.7  # Reduced learning
        else:
            balance_factor = 1.0

        # Principle 3: Check for productive imbalance patterns
        # Example: Strong Metal->Water->Wood chain (generation cycle)
        gen_boost = 1.0 + 0.2 * self.cycle_pattern_strengths['generation']

        # Conquest cycle modifies learning rate downward
        con_damp = 1.0 - 0.15 * self.cycle_pattern_strengths['conquest']

        # Balance contribution
        bal_factor = 1.0 + 0.05 * (self.cycle_pattern_strengths['balance'] - 0.5)

        # Calculate final factor with bounded values
        harmony_factor = balance_factor * gen_boost * con_damp * bal_factor
        harmony_factor = max(0.5, min(1.5, harmony_factor))  # Keep within reasonable bounds

        # Store in history
        self.history.append(harmony_factor)
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return harmony_factor


class CyclicalElementScheduler:
    """
    Learning rate scheduler based on the Wu Xing five-element cycle.
    Follows the natural generation and conquest cycles of the five elements.
    """

    def __init__(self, optimizer, cycle_length=100, base_decay=0.95, min_lr=1e-6, max_lr=1e-2,
                 element_sequence=None):
        """
        Initialize CyclicalElementScheduler with improved strategy.

        Args:
            optimizer: Optimizer to schedule (already initialized)
            cycle_length: Length of a complete five-element cycle
            base_decay: Base learning rate decay factor
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            element_sequence: Custom element sequence (default: water->wood->fire->earth->metal)
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

        # Default element sequence follows generation cycle
        self.element_sequence = element_sequence or [
            Element.WATER,  # Start with Water
            Element.WOOD,  # Water generates Wood
            Element.FIRE,  # Wood generates Fire
            Element.EARTH,  # Fire generates Earth
            Element.METAL  # Earth generates Metal
        ]
        self.current_element = self.element_sequence[0]

        # Element-specific learning rate factors
        self.element_factors = {
            Element.WATER: 1.0,  # Balanced, flowing
            Element.WOOD: 1.2,  # Growth, expansion
            Element.FIRE: 1.5,  # Transformation, activity
            Element.EARTH: 0.8,  # Stability, consolidation
            Element.METAL: 0.6  # Refinement, precision
        }

        # Transition factors - boost or dampen during transitions
        self.transition_factors = {
            (Element.WATER, Element.WOOD): 1.1,  # Water->Wood: Natural growth
            (Element.WOOD, Element.FIRE): 1.2,  # Wood->Fire: Energetic transformation
            (Element.FIRE, Element.EARTH): 0.7,  # Fire->Earth: Cooling stabilization
            (Element.EARTH, Element.METAL): 0.8,  # Earth->Metal: Gradual refinement
            (Element.METAL, Element.WATER): 0.9  # Metal->Water: Collection & condensation
        }

        # History tracking
        self.lr_history = []

    def step(self, metrics=None):
        """Basic step with cycle"""
        self.step_count += 1

        # Calculate element phase position
        phase_length = self.cycle_length // len(self.element_sequence)
        cycle_position = self.step_count % self.cycle_length
        phase_index = cycle_position // phase_length

        # Calculate phase progress (0-1 within current phase)
        phase_progress = (cycle_position % phase_length) / phase_length

        # Get current and next elements
        current_element = self.element_sequence[phase_index]
        next_index = (phase_index + 1) % len(self.element_sequence)
        next_element = self.element_sequence[next_index]

        # Get element factors
        current_factor = self.element_factors[current_element]
        next_factor = self.element_factors[next_element]

        # Calculate transition factor
        transition_key = (current_element, next_element)
        transition_factor = self.transition_factors.get(transition_key, 1.0)

        # Apply smooth transition between elements
        if phase_progress > 0.8:  # Last 20% of phase - begin transition
            # Smoothly interpolate between current and next factor
            transition_progress = (phase_progress - 0.8) * 5  # Scale to 0-1
            interpolated_factor = (1 - transition_progress) * current_factor + transition_progress * next_factor
            # Apply transition boost/dampening
            element_factor = interpolated_factor * transition_factor
        else:
            element_factor = current_factor

        # Apply base decay
        decay_factor = self.base_decay ** (self.step_count / 100)

        # Combined factor
        combined_factor = element_factor * decay_factor

        # Update learning rates with bounds
        for i, group in enumerate(self.optimizer.param_groups):
            new_lr = max(self.min_lr, min(self.max_lr, self.base_lrs[i] * combined_factor))
            group['lr'] = new_lr

        # Store history
        self.lr_history.append(combined_factor)
        if len(self.lr_history) > 500:
            self.lr_history = self.lr_history[-500:]

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

        # Apply hexagram-based adjustment if available
        if 'hexagram_adjustment' in trackers:
            hexagram_adj = trackers['hexagram_adjustment']
            if hasattr(hexagram_adj, 'calculate_adjustment'):
                # Get adjustment factor
                adj_factor = hexagram_adj.calculate_adjustment()

                # Apply to learning rates
                for group in self.optimizer.param_groups:
                    group['lr'] = max(self.min_lr, min(self.max_lr, group['lr'] * adj_factor))

        return [group['lr'] for group in self.optimizer.param_groups]


class TaijiBatchNorm(nn.Module):
    """
    Batch normalization inspired by the Taiji (Yin-Yang) principle
    Adaptively balances between normalization and original distribution
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, balance_init=0.5):
        super(TaijiBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Running stats
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # Taiji balance parameter (learnable) - controls yin/yang balance
        self.yin_yang_balance = nn.Parameter(torch.ones(num_features) * balance_init)

        # Cycle history
        self.register_buffer('balance_history', torch.zeros(100, num_features))
        self.history_index = 0

    def forward(self, x):
        # IMPORTANT: This function needs to be careful with computational graphs
        if self.training:
            # Detach x for statistics calculation to avoid graph connections
            with torch.no_grad():
                batch_mean = x.detach().mean(dim=0)
                batch_var = x.detach().var(dim=0, unbiased=False)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # Get normalized value - still keeping graph connections for x
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

            # Apply Taiji principle
            balance = torch.sigmoid(self.yin_yang_balance).view(1, -1)

            # Store balance in history - use detached value
            with torch.no_grad():
                self.balance_history[self.history_index] = balance.squeeze().detach()
                self.history_index = (self.history_index + 1) % self.balance_history.size(0)

            # Apply the balanced normalization
            x = balance * x_normalized + (1 - balance) * x

            # Apply scale and shift
            return self.weight * x + self.bias
        else:
            # Inference mode
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

            # Apply fixed balance from training
            balance = torch.sigmoid(self.yin_yang_balance).view(1, -1)
            x = balance * x_normalized + (1 - balance) * x

            # Apply scale and shift
            return self.weight * x + self.bias

    def get_balance_cycle(self):
        """Get the cyclical balance pattern for visualization"""
        return self.balance_history.detach().numpy()


class BaguaActivations(nn.Module):
    """Enhanced activation functions based on the eight trigrams"""

    def __init__(self, adaptive=True):
        super(BaguaActivations, self).__init__()
        self.bagua = Bagua()
        self.adaptive = adaptive

        # Initialize activation strengths
        self.register_buffer('activation_strengths', torch.ones(8))

        # Track which trigram is active
        self.active_trigram = 'qian'

        # Map trigram index to name
        self.idx_to_trigram = {
            0: 'qian',
            1: 'kun',
            2: 'zhen',
            3: 'gen',
            4: 'kan',
            5: 'li',
            6: 'xun',
            7: 'dui'
        }

    def forward(self, x, trigram_name=None):
        """Apply trigram-specific activation"""
        # Detach x for safety - THIS WAS CAUSING THE BACKWARD GRAPH ISSUE
        # IMPORTANT: DO NOT DETACH x here as it breaks the computational graph
        # Instead, use a simpler approach

        if trigram_name is None:
            trigram_name = self.active_trigram

        # SIMPLIFIED APPROACH: Just use standard activations initially
        # This helps isolate if BaguaActivations is causing the issue
        return F.relu(x)

        # Original code (comment out for testing)
        """
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
        """

    def get_all_activations(self, x):
        """Apply all eight trigram activations and return results"""
        results = {}
        for trigram in self.bagua.trigrams.keys():
            results[trigram] = self.forward(x, trigram)
        return results

    def adaptive_forward(self, x, metrics=None):
        """Adaptively choose the best activation based on network metrics"""
        # SIMPLIFIED APPROACH: Just use ReLU for testing
        return F.relu(x)

        # Original code (comment out for testing)
        """
        if not self.adaptive or metrics is None:
            # Default to active trigram
            return self.forward(x)

        # Get accuracy trend and loss trend - ensure they are scalar values
        acc_increasing = bool(metrics.get('acc_increasing', False))
        loss_decreasing = bool(metrics.get('loss_decreasing', False))
        gradient_norm = float(metrics.get('gradient_norm', 1.0))

        # Choose trigram based on network state
        if loss_decreasing and acc_increasing:
            if gradient_norm > 1.0:
                # High improvement, high gradient - Fire (transformation)
                self.active_trigram = 'li'
            else:
                # High improvement, low gradient - Water (flow)
                self.active_trigram = 'kan'
        elif loss_decreasing and not acc_increasing:
            # Loss decreasing but acc not improving - Wind (gentle)
            self.active_trigram = 'xun'
        elif not loss_decreasing and acc_increasing:
            # Acc increasing but loss not decreasing - Thunder (movement)
            self.active_trigram = 'zhen'
        else:
            # No improvement - Earth (stability) or Mountain (stillness)
            self.active_trigram = 'gen' if gradient_norm < 0.5 else 'kun'

        # Apply the chosen activation
        return self.forward(x, self.active_trigram)
        """


class SimpleModel(nn.Module):
    """A neural network with DaoML components"""

    def __init__(self, input_size=9, hidden_sizes=[64, 32, 16], output_size=1,
                 use_taiji=True, use_bagua=True, use_loshu=False):
        super(SimpleModel, self).__init__()

        self.use_taiji = use_taiji
        self.use_bagua = use_bagua
        self.use_loshu = use_loshu

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        # Apply Lo Shu weight modulation if enabled
        if self.use_loshu and input_size == 9 and hidden_sizes[0] == 9:
            self.register_buffer('loshu_matrix', create_positional_matrix(input_size, hidden_sizes[0]))
        else:
            self.register_buffer('loshu_matrix', None)

        # Use Taiji Batch Normalization if enabled
        if self.use_taiji:
            self.input_norm = TaijiBatchNorm(hidden_sizes[0])
        else:
            self.input_norm = nn.BatchNorm1d(hidden_sizes[0])

        # Use Bagua Activations if enabled
        if self.use_bagua:
            self.activations = BaguaActivations(adaptive=True)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

            if self.use_taiji:
                self.hidden_norms.append(TaijiBatchNorm(hidden_sizes[i + 1]))
            else:
                self.hidden_norms.append(nn.BatchNorm1d(hidden_sizes[i + 1]))

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x, metrics=None):
        # IMPORTANT: We do NOT detach x here, as that would break the computational graph
        # We need to maintain the graph connections for proper backpropagation

        # Input layer with Lo Shu if enabled
        if self.loshu_matrix is not None:
            # Apply Lo Shu positional significance
            w = self.input_layer.weight * self.loshu_matrix
            x = F.linear(x, w, self.input_layer.bias)
        else:
            x = self.input_layer(x)

        # Apply norm and activation - handle the batch size=1 case
        if x.dim() == 2 and x.size(0) > 1:
            x = self.input_norm(x)

        # Apply activation - SIMPLIFIED for testing
        if self.use_bagua:
            # Simplified to avoid potential graph issues
            x = F.relu(x)
        else:
            x = F.relu(x)

        # Hidden layers
        for i, (layer, norm) in enumerate(zip(self.hidden_layers, self.hidden_norms)):
            x = layer(x)
            if x.dim() == 2 and x.size(0) > 1:
                x = norm(x)

            # Apply activation - simplified
            x = F.relu(x)

        # Output layer
        return self.output_layer(x)


class DaoMLTrainer:
    """Complete training system with all Daoist components integrated"""

    def __init__(self, model, optimizer=None, lr=0.001, wu_wei_ratio=0.7):
        self.model = model

        # Initialize trackers
        self.hexagram_tracker = IChingStateTracker()
        self.wu_xing_system = WuXingSystem()

        # Initialize enhancers
        self.transition_advantage = IChingTransitionAdvantage(self.hexagram_tracker)
        self.wu_xing_harmony = WuXingLRHarmony(self.wu_xing_system)

        # Create optimizer if not provided
        if optimizer is None:
            model_params = list(model.parameters())
            self.optimizer = ImprovedWuWeiOptimizer(
                model_params, lr=lr, wu_wei_ratio=wu_wei_ratio,
                use_momentum_guidance=True, harmonic_update=True
            )
        else:
            self.optimizer = optimizer

        # Create learning rate scheduler
        self.scheduler = CyclicalElementScheduler(self.optimizer, cycle_length=100)

        # Metrics tracking
        self.metrics = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'hexagrams': [], 'elements': [],
            'lr_adjustments': [],
            'batch_metrics': []
        }

        # Batch-level metrics for adaptive components
        self.current_batch_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'loss_decreasing': False,
            'acc_increasing': False,
            'gradient_norm': 0.0,
            'activation_mean': 0.0,
            'activation_std': 0.0,
            'weight_norm': 0.0
        }

        # Debug flag for isolating backward graph issues
        self.debug_mode = True

    def train_epoch(self, train_loader, val_loader=None):
        """Train for one epoch with full Dao integration"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Previous metrics for comparison
        prev_loss = float('inf') if len(self.metrics['train_loss']) == 0 else self.metrics['train_loss'][-1]
        prev_acc = 0.0 if len(self.metrics['train_acc']) == 0 else self.metrics['train_acc'][-1]

        # Training loop
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # IMPORTANT: Make sure inputs and targets are detached from any previous computation
            inputs = inputs.clone()  # Clone to ensure it's a fresh tensor
            targets = targets.clone()

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass - SIMPLIFIED for debugging
            if self.debug_mode:
                # Simple forward pass without metrics or custom components
                outputs = self.model(inputs)
            else:
                # Normal forward pass with metrics
                outputs = self.model(inputs, self.current_batch_metrics)

            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(outputs, targets)

            # Backward pass - with retain_graph for debugging
            if self.debug_mode:
                loss.backward(retain_graph=True)  # Diagnostic approach to isolate issue
            else:
                loss.backward()

            # Calculate network metrics AFTER backward to avoid graph connections
            with torch.no_grad():
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                batch_correct = (predicted == targets).sum().item()
                batch_total = targets.size(0)
                batch_acc = batch_correct / batch_total

                # Calculate gradient and weight norms
                grad_norm = 0.0
                weight_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
                    weight_norm += p.norm().item()

                # Safe update of batch metrics with only primitive values
                batch_loss = float(loss.item())
                self.current_batch_metrics = {
                    'loss': batch_loss,
                    'accuracy': float(batch_acc),
                    'loss_decreasing': bool(batch_idx > 0 and batch_loss < self.current_batch_metrics['loss']),
                    'acc_increasing': bool(batch_idx > 0 and batch_acc > self.current_batch_metrics['accuracy']),
                    'gradient_norm': float(grad_norm),
                    'activation_mean': float(outputs.mean().item()),
                    'activation_std': float(outputs.std().item()),
                    'weight_norm': float(weight_norm)
                }

            # Store batch metrics for history
            self.metrics['batch_metrics'].append(self.current_batch_metrics.copy())
            if len(self.metrics['batch_metrics']) > 1000:
                self.metrics['batch_metrics'] = self.metrics['batch_metrics'][-1000:]

            # Update trackers AFTER backward and with no_grad to prevent graph issues
            with torch.no_grad():
                self.hexagram_tracker.update_batch(self.model, self.current_batch_metrics)
                self.wu_xing_system.update_batch(self.model, self.current_batch_metrics)

                # Calculate adjustments
                hexagram_adjustment = self.transition_advantage.calculate_adjustment()
                harmony_adjustment = self.wu_xing_harmony.get_harmony_factor()
                combined_adjustment = hexagram_adjustment * harmony_adjustment

                # Apply learning rate adjustment
                for group in self.optimizer.param_groups:
                    group['lr'] = group['lr'] * combined_adjustment

            # Optimizer step
            if self.debug_mode:
                # Simple optimizer step for debugging
                self.optimizer.step()
            else:
                # Full optimizer step with trackers
                trackers = {
                    'hexagram_state': self.hexagram_tracker,
                    'wu_xing_balance': self.wu_xing_system,
                    'hexagram_adjustment': self.transition_advantage
                }

                if hasattr(self.optimizer, 'step_with_trackers'):
                    self.optimizer.step_with_trackers(trackers)
                else:
                    self.optimizer.step()

            # Update metrics
            total_loss += batch_loss * batch_total
            correct += batch_correct
            total += batch_total

            # Record current state periodically
            if batch_idx % 10 == 0:
                current_hex = self.hexagram_tracker.get_current_hexagram_index()
                dominant_element = self.wu_xing_system.get_dominant_element()
                self.metrics['hexagrams'].append((batch_idx, current_hex))
                self.metrics['elements'].append((batch_idx, dominant_element))
                self.metrics['lr_adjustments'].append((batch_idx, combined_adjustment))

                # Diagnostic print for debugging
                if self.debug_mode and batch_idx % 50 == 0:
                    print(
                        f"Batch {batch_idx}: Loss {batch_loss:.4f}, Acc {batch_acc:.4f}, Hex {current_hex}, Element {dominant_element}")

        # Calculate epoch metrics
        epoch_loss = total_loss / max(1, total)
        epoch_acc = correct / max(1, total)

        # Update epoch-level metrics
        self.metrics['train_loss'].append(epoch_loss)
        self.metrics['train_acc'].append(epoch_acc)

        # Update success rates in optimizer
        if hasattr(self.optimizer, 'update_success_rates'):
            loss_improvement = prev_loss - epoch_loss
            self.optimizer.update_success_rates(loss_improvement)

        # Update scheduler with metrics
        if hasattr(self.scheduler, 'update_with_metrics'):
            self.scheduler.update_with_metrics(
                epoch_loss,
                epoch_acc,
                {
                    'wu_xing_balance': self.wu_xing_system,
                    'hexagram_adjustment': self.transition_advantage
                }
            )
        else:
            self.scheduler.step()

        # Validation if provided
        if val_loader is not None:
            val_loss, val_acc = self.validate(val_loader)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_acc'].append(val_acc)

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """Evaluate model on validation data"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                # Forward pass
                outputs = self.model(inputs)
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(outputs, targets)

                # Calculate accuracy
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                batch_correct = (predicted == targets).sum().item()
                batch_total = targets.size(0)

                # Update metrics
                val_loss += loss.item() * batch_total
                correct += batch_correct
                total += batch_total

        # Calculate average metrics
        avg_loss = val_loss / max(1, total)
        accuracy = correct / max(1, total)

        return avg_loss, accuracy

    def get_metrics(self):
        """Get all training metrics"""
        return self.metrics

    def get_current_state(self):
        """Get current state of I-Ching and Wu Xing systems"""
        state = {
            'hexagram': {
                'index': self.hexagram_tracker.get_current_hexagram_index(),
                'name': self.hexagram_tracker.get_current_hexagram_name(),
                'lines': self.hexagram_tracker.get_current_hexagram_lines()
            },
            'elements': self.wu_xing_system.get_element_strengths(),
            'dominant_element': self.wu_xing_system.get_dominant_element(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'wu_wei_ratio': self.optimizer.param_groups[0].get('wu_wei_ratio', None)
        }
        return state

    def visualize_training(self):
        """Visualize training metrics"""
        metrics = self.metrics

        # Create a figure with multiple plots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))

        # Plot losses
        axs[0, 0].plot(metrics['train_loss'], label='Train Loss')
        if metrics['val_loss']:
            axs[0, 0].plot(metrics['val_loss'], label='Validation Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].set_title('Loss Curves')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        # Plot accuracies
        axs[0, 1].plot(metrics['train_acc'], label='Train Accuracy')
        if metrics['val_acc']:
            axs[0, 1].plot(metrics['val_acc'], label='Validation Accuracy')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].set_title('Accuracy Curves')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)

        # Plot learning rate adjustments
        if 'lr_adjustments' in metrics and metrics['lr_adjustments']:
            steps = [adj[0] for adj in metrics['lr_adjustments']]
            adjustments = [adj[1] for adj in metrics['lr_adjustments']]

            axs[1, 0].plot(steps, adjustments)
            axs[1, 0].set_xlabel('Batch')
            axs[1, 0].set_ylabel('Learning Rate Adjustment')
            axs[1, 0].set_title('Learning Rate Harmony Adjustments')
            axs[1, 0].grid(True, alpha=0.3)

        # Plot hexagram transitions
        if 'hexagrams' in metrics and metrics['hexagrams']:
            steps = [hex_info[0] for hex_info in metrics['hexagrams']]
            hexagrams = [hex_info[1] for hex_info in metrics['hexagrams']]

            axs[1, 1].plot(steps, hexagrams, 'o-', markersize=4)
            axs[1, 1].set_xlabel('Batch')
            axs[1, 1].set_ylabel('Hexagram Index')
            axs[1, 1].set_title('I-Ching Hexagram States')
            axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Plot Wu Xing element strengths
        if 'elements' in metrics and metrics['elements']:
            plt.figure(figsize=(12, 6))

            # Extract element data
            elements_data = {}
            for element in Element:
                elements_data[element] = []

            steps = []

            for i, (step, element) in enumerate(metrics['elements']):
                if i % 5 == 0:  # Sample to reduce density
                    steps.append(step)
                    # Get element strengths at this point
                    strengths = self.wu_xing_system.get_element_strengths()
                    for element, strength in strengths.items():
                        elements_data[element].append(strength)

            # Plot each element
            for element, strengths in elements_data.items():
                if len(strengths) == len(steps):
                    plt.plot(steps, strengths, label=str(element), color=element.color)

            plt.xlabel('Batch')
            plt.ylabel('Element Strength')
            plt.title('Wu Xing Element Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()


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


def main():
    """Main function demonstrating DaoML framework with diagnostic approach"""
    print("=== DaoML Framework Example ===")

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X_train, y_train, X_test, y_test = generate_data(n_samples=3000, n_features=9)
    print(
        f"Data shapes: X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}, y_test {y_test.shape}")

    # Create dataset loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # DIAGNOSTIC APPROACH: Start with simplified components
    # First test a plain model with standard optimizer
    print("\nSTEP 1: Testing with standard model and optimizer...")
    standard_model = nn.Sequential(
        nn.Linear(9, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

    optimizer = torch.optim.Adam(standard_model.parameters(), lr=0.01)

    # Test a few batches with standard training loop
    standard_model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = standard_model(inputs)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, targets)
        loss.backward()  # This should work fine
        optimizer.step()

        if batch_idx >= 2:  # Just test a few batches
            break

    print("Standard model test passed.")

    # STEP 2: Test with SimpleModel but no custom components
    print("\nSTEP 2: Testing with SimpleModel but disabled custom components...")
    simple_model = SimpleModel(
        input_size=9,
        hidden_sizes=[64, 32, 16],
        output_size=1,
        use_taiji=False,  # Disable custom components for now
        use_bagua=False,
        use_loshu=False
    )

    simple_optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)

    # Test a few batches with this model
    simple_model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        simple_optimizer.zero_grad()
        outputs = simple_model(inputs)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, targets)
        loss.backward()  # This should also work fine
        simple_optimizer.step()

        if batch_idx >= 2:  # Just test a few batches
            break

    print("SimpleModel test passed.")

    # STEP 3: Test with DaoMLTrainer but simplified components
    print("\nSTEP 3: Testing with DaoMLTrainer but simplified model...")
    trainer_model = SimpleModel(
        input_size=9,
        hidden_sizes=[64, 32, 16],
        output_size=1,
        use_taiji=False,
        use_bagua=False,
        use_loshu=False
    )

    trainer = DaoMLTrainer(
        model=trainer_model,
        lr=0.01,
        wu_wei_ratio=0.7
    )

    # Set debug mode to use simplified training approach
    trainer.debug_mode = True

    # Try just one or two batches to test
    print("Testing DaoMLTrainer with a few batches...")
    try:
        # Process just a couple of batches for testing
        batch_iter = iter(train_loader)
        inputs, targets = next(batch_iter)

        # Reset gradients
        trainer.optimizer.zero_grad()

        # Forward pass - simplified in debug mode
        outputs = trainer.model(inputs)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, targets)

        # Backward pass with retain_graph for diagnostic purposes
        loss.backward(retain_graph=True)  # Use retain_graph=True for testing

        # Optimizer step
        trainer.optimizer.step()

        print("First batch succeeded.")

        # Try another batch
        inputs, targets = next(batch_iter)
        trainer.optimizer.zero_grad()
        outputs = trainer.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
        trainer.optimizer.step()

        print("Second batch succeeded. DaoMLTrainer initial test passed.")
    except Exception as e:
        print(f"Error in DaoMLTrainer test: {e}")
        import traceback
        traceback.print_exc()

    # STEP 4: If previous tests pass, try a full epoch with simplified setup
    if not hasattr(trainer, 'error_occurred'):
        print("\nSTEP 4: Testing one full epoch with simplified DaoMLTrainer...")
        try:
            train_loss, train_acc = trainer.train_epoch(train_loader)
            print(f"Full epoch training succeeded. Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        except Exception as e:
            print(f"Error in full epoch training: {e}")
            import traceback
            traceback.print_exc()

    # STEP 5: If everything passes, try with full components enabled
    # If we get this far, we can gradually enable the DaoML components
    print("\nSTEP 5: Testing with enabled components one by one...")

    # Test with Lo Shu enabled
    print("Testing with Lo Shu enabled...")
    loshu_model = SimpleModel(
        input_size=9,
        hidden_sizes=[64, 32, 16],
        output_size=1,
        use_taiji=False,
        use_bagua=False,
        use_loshu=True  # Enable Lo Shu
    )

    loshu_trainer = DaoMLTrainer(
        model=loshu_model,
        lr=0.01,
        wu_wei_ratio=0.7
    )
    loshu_trainer.debug_mode = True

    # Try processing a couple of batches
    try:
        batch_iter = iter(train_loader)
        inputs, targets = next(batch_iter)

        # Forward and backward
        loshu_trainer.optimizer.zero_grad()
        outputs = loshu_trainer.model(inputs)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
        loshu_trainer.optimizer.step()

        print("Lo Shu test passed.")
    except Exception as e:
        print(f"Error with Lo Shu enabled: {e}")

    # Finally, only proceed to full training if tests passed
    print("\nSTEP 6: Proceeding with full training...")

    # Create the final model with components enabled based on test results
    final_model = SimpleModel(
        input_size=9,
        hidden_sizes=[64, 32, 16],
        output_size=1,
        use_taiji=False,  # Initially disabled
        use_bagua=False,  # Initially disabled
        use_loshu=True  # Enable if test passed
    )

    final_trainer = DaoMLTrainer(
        model=final_model,
        lr=0.01,
        wu_wei_ratio=0.7
    )
    final_trainer.debug_mode = True  # Start with debug mode enabled

    # Training settings
    epochs = 10  # Reduced for testing

    # Training loop
    print("\nStarting training with DaoML system...")
    for epoch in range(epochs):
        try:
            # Train one epoch
            train_loss, train_acc = final_trainer.train_epoch(train_loader, val_loader)

            # Get current state
            state = final_trainer.get_current_state()

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {final_trainer.metrics['val_loss'][-1]:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {final_trainer.metrics['val_acc'][-1]:.4f}")
        except Exception as e:
            print(f"Error in epoch {epoch + 1}: {e}")
            break

    print("\nTraining complete!")

    # Visualize results if training succeeded
    if len(final_trainer.metrics['train_loss']) > 0:
        print("\nVisualizing training results...")
        final_trainer.visualize_training()

    print("\n=== DaoML Example Complete ===")


if __name__ == "__main__":
    main()