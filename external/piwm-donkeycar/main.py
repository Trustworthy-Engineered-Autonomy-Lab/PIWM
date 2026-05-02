import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class BicycleModel:
    '''
    Constructor for bicyle model. Requires wheelbase L and constant velocity v
    '''
    def __init__(self, L : float, wheel_turn_speed : float):
        self.L = L
        self.wheel_angle = 0
        # how quickly the wheel can turn (deg/s)
        self.wheel_turn_speed = wheel_turn_speed

    '''
    Given a control input with steering angle phi, velocity v, predicts displacement in position and heading from our original position after delta time
    Returns x,y,heading
    Referenced: https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html#id1
    '''
    def predict(self, steering : float, v : float, delta: float):
        self.wheel_angle = self.move_towards(steering, self.wheel_angle, delta * self.wheel_turn_speed)
        d_heading = v * np.tan(self.wheel_angle) * delta / self.L
        d_x = v * np.cos(d_heading) * delta
        d_y = v * np.sin(d_heading) * delta
        return d_x, d_y, np.rad2deg(d_heading)
    
    def move_towards(self, target, value, delta):
        if value < target:
            value = min(target, value + delta)
        elif value > target:
            value = max(target, value - delta)
        return value
'''
rotates an xy vector by angle (in radians)
'''
def rotate_vector(vector : np.array, angle: float):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    return np.dot(rotation_matrix, vector)

def vector_to_components(vector : np.array):
    u = np.cos(np.deg2rad(vector))
    v = np.sin(np.deg2rad(vector))

    return (u, v)

def numpy_mse(vector : np.array):
    squared_vector = vector**2
    return np.mean(squared_vector)

def numpy_rmse(vector : np.array):
    return np.sqrt(numpy_mse(vector))

'''
takes two world space states and converts one into the relative coordinate system of the other

reference_state: the state we're using as our reference frame. From it's perspective, it's located
at [0,0] with yaw angle [0]

state: the state we want to transform from the world space to the reference_state frame.

Returns rel_state, the coordinates of state in reference state's frame

States are nd.array types with data: [pos_x, pos_y, yaw_angle]
'''
def global_state_to_relative(reference_state : np.array, state : np.array) -> np.array:
    rel_state = np.zeros(shape=(3))

    # if our states are in the same place, they must be [0,0,0] relative to themselves
    if reference_state == state:
        return rel_state

    displacement = state[0:2] - reference_state[0:2]
    # rotate displacement vector to car's reference frame by negating heading
    displacement = rotate_vector(displacement, np.deg2rad(-reference_state[2]))

    # calculate relative heading
    heading = state[2] - reference_state[2]

    rel_state[0:2] = displacement
    rel_state[2] = heading

    return rel_state

def predict_dataset(model: BicycleModel, data: list):
    error = np.zeros(shape=(len(data), 3))
    batch_error = np.zeros(shape=(len(data[0]), 3))

    for batch in data:
        # ignore point zero since it's always zero
        for i in range (1, len(batch)):
            point = batch[i]
            # pred = model.predict(point[1][0], point[1][1], point[])



def main():
    # TODO need to add ability to read actions from car in npz
    npz = np.load("rel_traj2.npz", allow_pickle=True)
    data = list(npz['data'])
    npz.close()

if __name__ == "__main__":
    main()