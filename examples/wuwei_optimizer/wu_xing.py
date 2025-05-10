import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, List, Tuple, Optional
import networkx as nx

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
            Element.WATER: Element.WOOD,   # Water nourishes Wood
            Element.WOOD: Element.FIRE,    # Wood feeds Fire
            Element.FIRE: Element.EARTH,   # Fire creates Earth (ash)
            Element.EARTH: Element.METAL,  # Earth bears Metal
            Element.METAL: Element.WATER   # Metal collects Water (condensation)
        }

        # Define conquest cycle (克, ke): each element conquers another
        self.conquest_cycle = {
            Element.WATER: Element.FIRE,   # Water extinguishes Fire
            Element.FIRE: Element.METAL,   # Fire melts Metal
            Element.METAL: Element.WOOD,   # Metal cuts Wood
            Element.WOOD: Element.EARTH,   # Wood breaks Earth
            Element.EARTH: Element.WATER   # Earth absorbs Water
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
            'generation': 0.3,    # Generation effect strength
            'conquest': 0.4,      # Conquest effect strength
            'insult': 0.3,        # Insult effect strength
            'mother_child': 0.2,  # Mother-child effect strength
            'control': 0.4,       # Control effect strength
            'rebellion': 0.3,     # Rebellion effect strength
            'over_acting': 0.3,   # Over-acting effect strength
            'under_acting': 0.3,  # Under-acting effect strength
            'balance': 0.15       # Balance effect strength
        }

        # Thresholds for various cycles
        self.thresholds = {
            'insult': 2.5,        # Threshold ratio for insult cycle
            'control': 1.8,       # Dominance threshold for control cycle
            'rebellion': 1.5,     # Weakening threshold for rebellion cycle
            'over_acting': 2.0,   # Threshold for over-acting cycle
            'under_acting': 0.4,  # Threshold for under-acting cycle
            'balance': 0.25       # Imbalance threshold for balance cycle (as fraction of average)
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
                effect = -self.coefficients['control'] * (self.state[source]**2) / self.state[target]
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
                effect = self.coefficients['rebellion'] * self.state[source] * self.state[target_conquerer] / self.state[target]
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
                effect = self.coefficients['over_acting'] * (self.state[source] - self.thresholds['over_acting'] * self.state[target])
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
                effect = -self.coefficients['under_acting'] * (self.thresholds['under_acting'] * self.state[target] - self.state[source])
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

    def run_simulation(self, steps: int = 100) -> List[Dict[Element, float]]:
        """
        Run simulation for multiple steps.

        Args:
            steps: Number of time steps

        Returns:
            History of system states
        """
        for _ in range(steps):
            self.step()

        return self.history

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

    def plot_history(self):
        """Plot the history of element values"""
        history_array = np.array([[state[element] for element in Element]
                                 for state in self.history])

        plt.figure(figsize=(12, 6))

        for i, element in enumerate(Element):
            plt.plot(history_array[:, i],
                     label=str(element),
                     color=element.color,
                     linewidth=2)

        plt.title('Complete Wu Xing System Dynamics')
        plt.xlabel('Time Step')
        plt.ylabel('Element Strength')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_cycle_effects(self):
        """Plot the effects of each cycle on the elements"""
        cycles = list(self.cycle_effects.keys())
        elements = list(Element)

        # Create a figure with subplots for each cycle
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        # Plot each cycle's effects
        for i, cycle in enumerate(cycles):
            if i < len(axes):
                ax = axes[i]

                # Get effects for this cycle
                effects = [self.cycle_effects[cycle][element] for element in elements]
                colors = [element.color for element in elements]

                # Create bar chart
                bars = ax.bar(range(len(elements)), effects, color=colors, alpha=0.7)

                # Add element names as x-tick labels
                ax.set_xticks(range(len(elements)))
                ax.set_xticklabels([str(element) for element in elements], rotation=45)

                # Add title and grid
                ax.set_title(f'{cycle.capitalize()} Cycle Effects')
                ax.grid(True, alpha=0.3)

                # Add zero line
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.show()

    def visualize_system_state(self, state_idx: int = -1):
        """
        Visualize the current system state with all relationships.

        Args:
            state_idx: Index of state to visualize (-1 for current)
        """
        state = self.history[state_idx]

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes (elements)
        for element in Element:
            G.add_node(str(element), weight=state[element])

        # Add edges for all relationships
        # Generation cycle (green edges)
        for source, target in self.generation_cycle.items():
            G.add_edge(str(source), str(target),
                      relationship="generation",
                      weight=2.0,
                      color="green")

        # Conquest cycle (red edges)
        for source, target in self.conquest_cycle.items():
            G.add_edge(str(source), str(target),
                      relationship="conquest",
                      weight=1.5,
                      color="red")

        # Calculate positions (pentagon layout)
        pos = {
            "Water": (0, -1),    # Bottom
            "Wood": (1, 0),      # Right
            "Fire": (0.5, 1),    # Top right
            "Earth": (-0.5, 1),  # Top left
            "Metal": (-1, 0)     # Left
        }

        # Get edge colors and weights
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

        # Get node sizes based on element strengths
        node_sizes = [state[Element[node.upper()]] * 20 for node in G.nodes()]

        # Get node colors
        node_colors = [Element[node.upper()].color for node in G.nodes()]

        # Create figure
        plt.figure(figsize=(12, 10))

        # Draw the graph
        nx.draw_networkx(
            G, pos=pos,
            with_labels=True,
            node_size=node_sizes,
            node_color=node_colors,
            font_size=12,
            font_weight='bold',
            edge_color=edge_colors,
            width=edge_weights,
            arrowsize=20,
            alpha=0.8
        )

        # Add element values as labels
        for element in Element:
            x, y = pos[str(element)]
            plt.text(x, y-0.15, f"{state[element]:.1f}",
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3",
                             facecolor='white',
                             alpha=0.7))

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Generation'),
            Line2D([0], [0], color='red', lw=2, label='Conquest')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        # Add title
        if state_idx == -1:
            plt.title('Current Wu Xing System State')
        else:
            plt.title(f'Wu Xing System State at Step {state_idx}')

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_cycle_network(self):
        """Visualize all cycles as a network"""
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes (elements)
        for element in Element:
            G.add_node(str(element))

        # Add edges for all relationships with different colors
        # Generation cycle (green edges)
        for source, target in self.generation_cycle.items():
            G.add_edge(str(source), str(target),
                      relationship="generation",
                      weight=2.0,
                      color="green")

        # Conquest cycle (red edges)
        for source, target in self.conquest_cycle.items():
            G.add_edge(str(source), str(target),
                      relationship="conquest",
                      weight=1.5,
                      color="red")

        # Insult cycle (orange edges)
        for target, source in self.conquest_cycle.items():
            G.add_edge(str(source), str(target),
                      relationship="insult",
                      weight=1.0,
                      color="orange",
                      style="dashed")

        # Control cycle (purple edges)
        for source, target in self.conquest_cycle.items():
            G.add_edge(str(source), str(target),
                      relationship="control",
                      weight=1.0,
                      color="purple",
                      style="dotted")

        # Calculate positions (pentagon layout)
        pos = {
            "Water": (0, -1),    # Bottom
            "Wood": (1, 0),      # Right
            "Fire": (0.5, 1),    # Top right
            "Earth": (-0.5, 1),  # Top left
            "Metal": (-1, 0)     # Left
        }

        # Create figure
        plt.figure(figsize=(14, 12))

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos=pos,
            node_size=2000,
            node_color=[Element[node.upper()].color for node in G.nodes()],
            alpha=0.8
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos=pos,
            font_size=12,
            font_weight='bold'
        )

        # Draw edges for each relationship type
        for relationship, color, style in [
            ("generation", "green", "solid"),
            ("conquest", "red", "solid"),
            ("insult", "orange", "dashed"),
            ("control", "purple", "dotted")
        ]:
            # Get edges of this relationship type
            edges = [(u, v) for u, v, d in G.edges(data=True) if d['relationship'] == relationship]

            # Draw these edges
            nx.draw_networkx_edges(
                G, pos=pos,
                edgelist=edges,
                width=2,
                edge_color=color,
                style=style,
                arrowsize=20,
                alpha=0.7,
                connectionstyle="arc3,rad=0.2"  # Curved edges
            )

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Generation (生 Sheng)'),
            Line2D([0], [0], color='red', lw=2, label='Conquest (克 Ke)'),
            Line2D([0], [0], color='orange', lw=2, linestyle='dashed', label='Insult (侮 Wu)'),
            Line2D([0], [0], color='purple', lw=2, linestyle='dotted', label='Control (乘 Cheng)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        # Add title
        plt.title('Complete Wu Xing Cycle Network')

        plt.axis('off')
        plt.tight_layout()
        plt.show()


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
        # Adjust element strengths based on network metrics

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
        """
        Update the Wu Xing system state based on epoch metrics

        Args:
            model: The neural network model
            network_metrics: Dictionary of network metrics
        """
        # Similar to update_batch but with larger effects for epoch-level updates

        # Water (adaptability) strengthened by decreasing loss, varied gradients
        if network_metrics['loss_decreasing']:
            self.state[Element.WATER] += 1.0
        if network_metrics['gradient_norm'] > 0.5:
            self.state[Element.WATER] += 0.5

        # Wood (growth) strengthened by increasing accuracy
        if network_metrics['accuracy'] > 0.7:
            self.state[Element.WOOD] += 1.0

        # Fire (transformation) strengthened by high gradient activity
        if network_metrics['gradient_norm'] > 1.0:
            self.state[Element.FIRE] += 1.5

        # Earth (stability) strengthened by balanced metrics
        if 0.7 < network_metrics['accuracy'] < 0.9:
            self.state[Element.EARTH] += 1.0
        if 0.5 < network_metrics['weight_norm'] < 2.0:
            self.state[Element.EARTH] += 0.5

        # Metal (precision) strengthened by high accuracy, low gradients
        if network_metrics['accuracy'] > 0.9:
            self.state[Element.METAL] += 1.5
        if network_metrics['gradient_norm'] < 0.1:
            self.state[Element.METAL] += 0.5

        # Apply system dynamics with a larger time step
        self.step(dt=0.1)