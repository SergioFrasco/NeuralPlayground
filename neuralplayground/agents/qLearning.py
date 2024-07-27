import numpy as np
from neuralplayground.agents import AgentCore

class QLearningAgent(AgentCore):
    def __init__(self, state_space, actions, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.9, exploration_decay=0.5, state_bins=10):
        super().__init__()
        self.state_space = state_space
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

        self.state_bins = state_bins
        self.q_table = np.zeros((self.state_bins, self.state_bins, len(actions)))

    def discretize_state(self, state):
        state = np.array(state)
        discretized = np.floor(state * self.state_bins).astype(int)
        discretized = np.clip(discretized, 0, self.state_bins - 1)
        return tuple(discretized)

    def act(self, state):
        state = self.discretize_state(state)
        if np.random.rand() < self.exploration_rate:
            action_index = np.random.choice(len(self.actions))
        else:
            action_index = np.argmax(self.q_table[state])
        return self.actions[action_index]

    def update_q_table(self, state, action, reward, next_state):
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)
        action_index = self.actions.index(action)

        best_next_action = np.max(self.q_table[next_state])
        self.q_table[state][action_index] = (1 - self.learning_rate) * self.q_table[state][action_index] + \
            self.learning_rate * (reward + self.discount_factor * best_next_action)
        
        print(self.q_table)
