# custom_arena.py
import numpy as np
from neuralplayground.arenas import DiscreteObjectEnvironment
from neuralplayground.experiments import SargoliniDataTrajectory

class Sutton1999Discrete(DiscreteObjectEnvironment):
    def __init__(self, **kwargs):
        super().__init__(recording_index=None, environment_name="CustomDiscrete", verbose=False, experiment_class=SargoliniDataTrajectory, **kwargs)
        self.custom_walls = self.create_custom_walls()
        self.wall_list = self.default_walls + self.custom_walls
    
    def create_custom_walls(self):
        custom_walls = []
        
        # Define walls as pairs of points: [[x1, y1], [x2, y2]]
        custom_walls.append(np.array([[0, -70], [0, -90]]))  
        custom_walls.append(np.array([[0, 70], [0, 90]])) 
        custom_walls.append(np.array([[0, 50], [0, -50]]))
        custom_walls.append(np.array([[-90, 10], [-70, 10]]))
        custom_walls.append(np.array([[-50, 10], [0, 10]]))
        custom_walls.append(np.array([[90, -10], [70, -10]]))
        custom_walls.append(np.array([[50, -10], [0, -10]]))
    
        return custom_walls

    def _create_default_walls(self):
        # Optionally, override this method if you want custom default walls
        super()._create_default_walls()
