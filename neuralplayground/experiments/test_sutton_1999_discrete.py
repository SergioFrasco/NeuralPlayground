# main_experiment.py
from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

import numpy as np
import matplotlib.pyplot as plt

from neuralplayground.experiments import SargoliniDataTrajectory
from neuralplayground.agents import DiscreteAgent
from neuralplayground.arenas import Sutton1999Discrete

# Initialize agent
agent = DiscreteAgent()

time_step_size = 0.1  # seconds
agent_step_size = 20

# Environment configuration
env_kwargs = {
    "arena_x_limits": [-90, 90],  # X-axis limits of the arena
    "arena_y_limits": [-90, 90],  # Y-axis limits of the arena
    "n_objects": 5,  # Number of distinct objects
    "state_density": 10,  # Density of the state grid
    "agent_step_size": 1,  # Size of the agent's step
    "use_behavioural_data": False,  # Assuming not using behavioral data for this example
    "data_path": None  # Path to data if behavioral data is used
}

# Create an instance of the custom environment
env = Sutton1999Discrete(**env_kwargs)

# Set default and custom walls (if not already set in the custom environment)
env._create_default_walls()

n_steps = 100  # Number of steps

# Initialize environment
obs = env.reset()

# Run a loop to see the agent in action
for _ in range(n_steps):
    action = agent.act(obs)
    obs, state, reward = env.step(action)
    print(f"Action: {action}, Observation: {obs}, State: {state}, Reward: {reward}")

# Plot the trajectory
ax = env.plot_trajectory()
ax.grid()
plt.show()
