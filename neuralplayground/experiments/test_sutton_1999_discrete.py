# main_experiment.py
from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

import numpy as np
import matplotlib.pyplot as plt

from neuralplayground.experiments import SargoliniDataTrajectory
from neuralplayground.agents import DiscreteAgent, QLearningAgent
from neuralplayground.arenas import Sutton1999Discrete



# Initialize environment and its variables
env_kwargs = {
    "arena_x_limits": [-90, 90],
    "arena_y_limits": [-90, 90],
    "n_objects": 5,
    "state_density": 10,
    "agent_step_size": 1,
    "use_behavioural_data": False,
    "data_path": None
}

env = Sutton1999Discrete(**env_kwargs)
env._create_default_walls()

agent = QLearningAgent(state_space=env.state_space, actions=env.actions)

time_step_size = 0.1
agent_step_size = 20
n_steps = 100

obs = env.reset()

for episode in range(n_steps):
    state = tuple(env.reset()[:2])  # Ensure state is a tuple of (x, y)
    done = False

    while not done:
        action = agent.act(state)
        observation, next_state, reward = env.step(action)
        next_state = tuple(next_state[:2])  # Ensure next_state is a tuple of (x, y)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state

    agent.exploration_rate *= agent.exploration_decay

ax = env.plot_trajectory()
ax.grid()
plt.show()
