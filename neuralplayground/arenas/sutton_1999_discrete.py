import numpy as np
from neuralplayground.arenas import DiscreteObjectEnvironment
from neuralplayground.experiments import SargoliniDataTrajectory

class Sutton1999Discrete(DiscreteObjectEnvironment):
    def __init__(self, **kwargs):
        super().__init__(recording_index=None, environment_name="CustomDiscrete", verbose=False, experiment_class=SargoliniDataTrajectory, **kwargs)
        self.custom_walls = self.create_custom_walls()
        self.wall_list = self.default_walls + self.custom_walls
        self.state_space = self._create_state_space()
        self.action_space = 4  # four possible actions: up, down, left, right
        self.actions = {
            0: np.array([0, 20]),    # up
            1: np.array([0, -20]),   # down
            2: np.array([-20, 0]),   # left
            3: np.array([20, 0])     # right
        }
    
    def create_custom_walls(self):
        custom_walls = []
        custom_walls.append(np.array([[0, -70], [0, -90]]))  
        custom_walls.append(np.array([[0, 70], [0, 90]])) 
        custom_walls.append(np.array([[0, 50], [0, -50]]))
        custom_walls.append(np.array([[-90, 10], [-70, 10]]))
        custom_walls.append(np.array([[-50, 10], [0, 10]]))
        custom_walls.append(np.array([[90, -10], [70, -10]]))
        custom_walls.append(np.array([[50, -10], [0, -10]]))
        return custom_walls

    def _create_default_walls(self):
        super()._create_default_walls()

    def _create_state_space(self):
        state_space = []
        for x in self.x_array:
            for y in self.y_array:
                state_space.append((x, y))
        return state_space
    
    def step(self, action: np.ndarray, normalize_step: bool = False, skip_every: int = 10):
        self.old_state = self.state.copy()
        if self.use_behavioral_data:
            if self.global_steps * skip_every >= self.experiment.position.shape[0] - 1:
                self.global_steps = np.random.choice(np.arange(skip_every))
            self.global_time = self.global_steps * self.time_step_size
            new_pos_state = (
                self.experiment.position[self.global_steps * skip_every, :],
                self.experiment.head_direction[self.global_steps * skip_every, :],
            )
            new_pos_state = np.concatenate(new_pos_state)
        else:
            if action[0] == 0:
                action_rev = np.array([0.0, -action[1]])    
            else:
                action_rev = action
            if normalize_step and np.linalg.norm(action) > 0:
                action_rev = action_rev / np.linalg.norm(action_rev)
                new_pos_state = self.state[-1] + self.agent_step_size * action_rev
            else:
                new_pos_state = self.state[-1] + action_rev
            new_pos_state, valid_action = self.validate_action(self.state[-1], action_rev, new_pos_state[:2])

        reward = self.reward_function(action, self.state[-1])
        observation = self.make_object_observation(new_pos_state)
        self.state = observation
        self.transition = {
            "action": action,
            "state": tuple(self.old_state[-1][:2]),  # Ensure state is a tuple of (x, y)
            "next_state": tuple(self.state[-1][:2]),  # Ensure next_state is a tuple of (x, y)
            "reward": reward,
            "step": self.global_steps,
        }
        self.history.append(self.transition)
        self._increase_global_step()
        return observation, tuple(self.state[-1][:2]), reward  # Ensure next_state is a tuple of (x, y)
