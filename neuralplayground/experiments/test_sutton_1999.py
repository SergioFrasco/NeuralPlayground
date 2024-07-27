from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

import numpy as np
import matplotlib.pyplot as plt
from neuralplayground.arenas import sutton_1999
from neuralplayground.agents import QLearningAgent

# Define action mappings
action_mappings = {
    'up': np.array([0, 10]),
    'down': np.array([0, -10]),
    'left': np.array([-10, 0]),
    'right': np.array([10, 0])
}

# Init environment
time_step_size = 0.1  # sec
agent_step_size = 3

env = sutton_1999(time_step_size=time_step_size, agent_step_size=agent_step_size)

# Define state space as continuous range normalized to [0, 1]
state_space = [(x / 100.0, y / 100.0) for x in range(-100, 101) for y in range(-100, 101)]
actions = ['up', 'down', 'left', 'right']

# Create the QLearningAgent
agent = QLearningAgent(state_space=state_space, actions=actions, state_bins=10)

# Number of steps to simulate
n_steps = 20000

# Initialize environment
obs, state = env.reset()
for i in range(n_steps):
    # Observe to choose an action
    action = agent.act(obs)
    # Convert action to numerical vector
    action_vector = action_mappings[action]
    # Run environment for given action
    obs, state, reward = env.step(action_vector)
    # Update Q-table based on action outcome
    agent.update_q_table(state, action, reward, obs)

    # Decay exploration rate
    agent.exploration_rate *= agent.exploration_decay
    print("exploration rate: ", agent.exploration_rate)

ax = env.plot_trajectory()
ax.grid()
plt.show()
