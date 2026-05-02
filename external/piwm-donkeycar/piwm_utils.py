import numpy as np
import pandas as pd
from tqdm import tqdm

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
    
    if np.all(reference_state == state):
        return rel_state

    displacement = state[0:2] - reference_state[0:2]
    # rotate displacement vector to car's reference frame by negating heading
    displacement = rotate_vector(displacement, np.deg2rad(-reference_state[2]))

    # calculate relative heading
    heading = state[2] - reference_state[2]

    rel_state[0:2] = displacement
    rel_state[2] = heading

    return rel_state


def slice_with_stride(arr, n, m):
    """
    Slice a NumPy array where each slice has n elements,
    and slices are spaced m elements apart.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
    n : int
        Number of elements per slice
    m : int
        Spacing between start of consecutive slices
    
    Returns:
    --------
    np.ndarray
        2D array where each row is a slice
    """
    # Calculate number of valid slices
    num_slices = (len(arr) - n) // m + 1
    
    # Create indices for all slices
    indices = np.arange(n) + np.arange(num_slices)[:, np.newaxis] * m
    
    return arr[indices]

def batch_to_relative(batch : np.array):
    rel_batch = batch.copy()
    start_time = batch[0][1]
    start_state = batch[0][2]
    # copy over values for each data point, applying transformations as needed
    for i in range(0, len(batch)):
        rel_batch[i][1] = batch[i][1] - start_time
        rel_batch[i][2] = global_state_to_relative(start_state, batch[i][2])
    return rel_batch
        

'''
(frame, timestamp, [posx, posy, yaw], [steering, throttle])
'''

def npz_to_relative_npz(npz, save_path, stride_window, stride_step, state_action_offset: int = 0):
    dataset = [(npz["frame"][i], npz["timestamps"][i], npz["state"][i][:3], npz["action"][i + state_action_offset], npz["state"][i]) for i in range(len(npz["timestamps"])-state_action_offset)]
    dataset = slice_with_stride(np.array(dataset, dtype=np.object_), stride_window, stride_step)
    
    i = 0
    while i < len(dataset):
        batch = dataset[i]
        if np.isnan(np.vstack(batch[:,2])).any().any():
            dataset = np.delete(dataset, i, axis=0)
            i -= 1
        i += 1
    rel_dataset = dataset

    for i, batch in enumerate(dataset):
        rel_batch = batch_to_relative(batch)
        rel_dataset[i] = rel_batch
    
    np.savez(save_path, data=rel_dataset)

def main():
    npz = dict(np.load("repaired_traj2(1).npz"))
    save_path = "rel_reparied_traj2.npz"
    stride_window = 10
    stride_step = 10

    npz_to_relative_npz(npz, save_path, stride_window, stride_step, 0)

if __name__ == "__main__":
    main()