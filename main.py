import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
Can you write a python program using pytorch to implement a Markov chain with a large number S (~1000) of states
conditioning the discrete distribution of an event among a large number of E (~1000) events.
The Markov chain transition matrix should be split in G (~10) groups of states.
Each group parametrized by:
- a probability (large = (1-sigma)) of staying in ones state (diagonal terms),
- a probability to change state within the group (off-diagonal terms randomized using a log-normal or gamma distribution scaled by a factor),
- and a probability to change state to a another group of states (off-diagonal terms randomized using a log-normal or gamma distribution scaled by a factor).
Among the G groups, there is a group of compromised states.
The distribution of events are different for each state, but one of the event, the neutral event (index 0) has a large probability (1-delta).
"""

class MarkovChainModel:
    def __init__(self, num_states=1000, num_groups=10, sigma=0.1, tau=0.1, compromised_group=0,
                 num_events=1000, delta=0.01, neutral_event=0):
        """
        Initialize Markov Chain Model
        
        Args:
            num_states (int): Number of states S (~1000)
            num_groups (int): Number of groups G (~10)
            sigma (float): Probability of leaving current state (diagonal terms = 1-sigma)
            tau (float): Probability of staying in current group (diagonal terms = 1-tau)
            compromised_group (int): Index of the compromised group
            num_events (int): Number of events E (~1000)
            delta (float): Probability of non-neutral events
            neutral_event (int): Index of the neutral event
        """
        self.S = num_states
        self.E = num_events  # Will be renamed to num_events
        self.G = num_groups
        self.sigma = sigma
        self.delta = delta
        self.tau = tau
        self.compromised_group = compromised_group
        self.neutral_event = neutral_event
        
        # Calculate states per group
        self.states_per_group = self.S // self.G
        
        # Initialize transition matrix and event distributions
        self.transition_matrix = self._create_transition_matrix()
        self.event_distributions = self._create_event_distributions()  # Renamed from element_distributions
    
    def _create_transition_matrix(self):
        # Initialize transition matrix
        P = torch.zeros((self.S, self.S))
        
        for g in range(self.G):
            start_idx = g * self.states_per_group
            end_idx = (g + 1) * self.states_per_group
            
            # Set diagonal terms (1-sigma)
            P[start_idx:end_idx, start_idx:end_idx] += torch.eye(self.states_per_group) * (1 - self.sigma)
            
            # Set intra-group transitions
            intra_group_probs = torch.distributions.LogNormal(-2, 1).sample((self.states_per_group, self.states_per_group))
            intra_group_probs = intra_group_probs * (1 - torch.eye(self.states_per_group))
            P[start_idx:end_idx, start_idx:end_idx] += intra_group_probs * (1 - self.tau) * self.sigma
            
            # Set inter-group transitions
            for other_g in range(self.G):
                if other_g != g:
                    other_start = other_g * self.states_per_group
                    other_end = (other_g + 1) * self.states_per_group
                    inter_group_probs = torch.distributions.LogNormal(-1, 1).sample((self.states_per_group, self.states_per_group))
                    P[start_idx:end_idx, other_start:other_end] = F.normalize(inter_group_probs, p=1, dim=1) * self.tau * self.sigma
        
        return P
    
    def _create_event_distributions(self):  # Renamed from _create_element_distributions
        # Initialize event distributions for each state
        distributions = torch.zeros((self.S, self.E))
        
        # Set high probability for neutral event (index 0)
        distributions[:, 0] = 1 - self.delta
        
        # Distribute remaining probability among other events
        other_probs = torch.distributions.LogNormal(-2, 1).sample((self.S, self.E-1))
        other_probs = F.normalize(other_probs, p=1, dim=1) * self.delta
        distributions[:, 1:] = other_probs
        
        return distributions

    def simulate_multiple(self, num_trajectories, num_steps, initial_state=0):
        """
        Simulate multiple trajectories of the Markov chain in parallel
        
        Args:
            num_trajectories (int): Number of trajectories to simulate
            num_steps (int): Number of steps per trajectory
            initial_state (int): Starting state for all trajectories
            
        Returns:
            tensor: Events tensor with shape (num_trajectories, num_steps) containing
                   the generated events for each trajectory at each time step
        """
        # Initialize tensor to store events only
        events = torch.zeros((num_trajectories, num_steps), dtype=torch.long)  # Renamed from elements
        
        # Initialize current states
        current_states = torch.full((num_trajectories,), initial_state, dtype=torch.long)
        
        # Simulate all trajectories in parallel
        for step in range(num_steps):
            # Sample events for current states
            event_probs = self.event_distributions[current_states]  # Renamed
            events[:, step] = torch.multinomial(event_probs, 1).squeeze()
            
            # Update states for next iteration
            transition_probs = self.transition_matrix[current_states]
            current_states = torch.multinomial(transition_probs, 1).squeeze()
        
        return events

def main():
    # Create and test the model
    model = MarkovChainModel(
        num_states=1000, num_groups=10, sigma=0.1, tau=0.1, compromised_group=0,
        num_events=1000, delta=0.05, neutral_event=0
    )
    
    # Simulate multiple trajectories
    num_trajectories = 1000
    num_steps = 1000
    events = model.simulate_multiple(num_trajectories, num_steps)
    
    print(f"Shape of events tensor: {events.shape}")
    print(f"First 10 events from first 10 time steps: {events[0:2, :]}")

    # Plot the first 10 events from the first 10 time steps
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(events[0:10, :], axis=1).T)
    plt.xlabel('Time Step')
    plt.ylabel('Trajectory')
    plt.title('First 10 Events from First 10 Time Steps')
    plt.show()

    plt.imshow(model.transition_matrix, aspect='auto', cmap='viridis')
    plt.show()

if __name__ == "__main__":
    main()
