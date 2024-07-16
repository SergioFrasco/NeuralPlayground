from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))
import numpy as np
import matplotlib.pyplot as plt

from neuralplayground.arenas import Simple2D, ConnectedRooms, sutton_1999

from neuralplayground.agents import RandomAgent, LevyFlightAgent


# Random agent generates a brownian motion. Levy flight is still experimental.
agent = LevyFlightAgent(step_size=0.8, scale=2.0, loc=0.0, beta=1.0, alpha=0.5, max_action_size=100)

time_step_size = 0.1 #seg
agent_step_size = 3

# # Init environment
# env = Simple2D(time_step_size = time_step_size,
#                agent_step_size = agent_step_size,
#                arena_x_limits=(-100, 100), 
#                arena_y_limits=(-100, 100))


# Init environment
env = sutton_1999(time_step_size = time_step_size,
                     agent_step_size = agent_step_size)

n_steps = 3000#30000

# Initialize environment
obs, state = env.reset()
for i in range(n_steps):
    # Observe to choose an action
    action = agent.act(obs)
    # Run environment for given action
    obs, state, reward = env.step(action)

ax = env.plot_trajectory()
ax.grid()
plt.show()