"""
Class building arena as
"Grid Cells Form a Global Representation of Connected Environments"
Carpenter et al. 2015
"""

import numpy as np

from .simple2d import Simple2D


class sutton_1999(Simple2D):
    """
    Simulated environment from https://doi.org/10.1016/j.cub.2015.02.037.
    Default parameters from experimental setting.
    """

    def __init__(
        self,
        environment_name: str = "sutton_1999",
        corridor_ysize: float = 90.0,
        singleroom_ysize: float = 90.0,
        singleroom_xsize: float = 90,
        door_size: float = 10.0,
        **env_kwargs,
    ):
        """
        Parameters
        ----------
        environment_name: str
            Name of the specific instantiation of the sutton_1999 class
        corridor_ysize : float
            corridor size from the paper, default 40.0 cm
        singleroom_ysize : float
            y-size of one of the rooms, default 90.0 cm
        singleroom_xsize : float
            x-size of one of the rooms, default 90.0 cm
        door_size : float
            door size from room to corridor, default 10 cm
        env_kwargs: dict
            time_step_size: float
                time_step_size * global_steps will give a measure of the time in the experimental setting
            agent_step_size: float
                Step size used when the action is a direction in x,y coordinate (normalize false in step())
                Agent_step_size * global_step_number will give a measure of the distance in the experimental setting
        """
        self.reward_map = {
            # Define specific rewards for certain states
            (-60, 60): 10.0,  # Example: State (x1, y1) has a reward of 10.0
            (60, 60): 5.0,   # Example: State (x2, y2) has a reward of 5.0
        }
        self.default_negative_reward = -1  # Slightly negative reward for other states

        self.corridor_ysize = corridor_ysize
        self.singleroom_ysize = singleroom_ysize
        self.singleroom_xsize = singleroom_xsize
        self.door_size = door_size

        env_kwargs["arena_x_limits"] = np.array([-self.singleroom_xsize, self.singleroom_xsize])
        env_kwargs["arena_y_limits"] = np.array([-self.singleroom_ysize, self.corridor_ysize])

        super().__init__(environment_name, **env_kwargs)

    def _create_custom_walls(self):
        self.custom_walls = []  # Create the list with custom_walls (needs to be created)
        self.custom_walls.append(np.array([[0, -70], [0, -self.singleroom_ysize]]))  
        self.custom_walls.append(np.array([[0, 70], [0, self.singleroom_ysize]])) 
        self.custom_walls.append(np.array([[0, 50], [0, -50]]))
        self.custom_walls.append(np.array([[-self.singleroom_xsize, 10], [-70, 10]]))
        self.custom_walls.append(np.array([[-50, 10], [0, 10]]))
        self.custom_walls.append(np.array([[self.singleroom_xsize,-10], [70, -10]]))
        self.custom_walls.append(np.array([[50, -10], [0, -10]]))
  
 
        # self.custom_walls.append(np.array([[25, 0], [25, -self.singleroom_ysize]]))    
        # self.custom_walls.append(np.array([[-self.singleroom_xsize, 0], [-25, 0]]))
        # self.custom_walls.append(np.array([[self.singleroom_xsize, 0], [25, 0]]))
    def step(self, action: None, normalize_step: bool = False):
        """Runs the environment dynamics. Increasing global counters.
        Given some action, return observation, new state and reward.

        Parameters
        ----------
        action: None or ndarray (2,)
            Array containing the action of the agent, in this case the delta_x and detla_y increment to position
        normalize_step: bool
            If true, the action is normalized to have unit size, then scaled by the agent step size

        Returns
        -------
        reward: float
            The reward that the animal receives in this state
        new_state: ndarray
            Update the state with the updated vector of coordinate x and y of position and head directions respectively
        observation: ndarray
            Array of the observation of the agent in the environment
        """
        if action is None:
            new_state = self.state
        else:
            if normalize_step:
                action = action / np.linalg.norm(action)
                new_state = self.state + self.agent_step_size * action
            else:
                new_state = self.state + action
            new_state, valid_action = self.validate_action(self.state, action, new_state)
            # If you get reward, it should be coded here
        self.state = np.asarray(new_state)
        observation = self.make_observation()
        self._increase_global_step()
        reward = self.reward_function(action, self.state)
        transition = {
            "action": action,
            "state": self.state,
            "next_state": new_state,
            "reward": reward,
            "step": self.global_steps,
        }
        self.history.append(transition)
        return observation, new_state, reward
        
    def reward_function(self, action, state):
        """Reward curriculum as a function of action, state
        and attributes of the environment

        Parameters
        ----------
        action:
            same as step method argument
        state:
            same as state attribute of the class

        Returns
        -------
        reward: float
            reward given the parameters
        """
        # Convert state to a hashable type if needed
        state_tuple = tuple(state)  # Assuming state is a list or ndarray

        # Check if the state is in the reward map
        if state_tuple in self.reward_map:
            reward = self.reward_map[state_tuple]
        else:
            reward = self.default_negative_reward
        
        return reward
