import argparse
from piwm_utils import npz_to_relative_npz
import numpy as np

'''
This is meant to be used as a command line utility for generating relative trajectory data from a global NPZ.
Given a global frame npz dataset input.npz, this program adds npz files ['rel_state'] and ['rel_window'] to the original npz.

The slicing of the original dataset into relative windows is controlled by the stride_window and stride_step parameters.
The stride_window is the number of datapoints included in a relative data batch.
The stride_step is the distance between the starts of data batches. When the stride_window and stride_step are equal, the data will be divided as evenly as possible over
floor(total_datapoints/stride_window) data batches.

Example usage:
python3 global_to_relative_npz.py input_path output_path stride_window stride_step
python3 global_to_relative_npz.py my_input.npz my_output.npz 5 5
'''

def main(args):
    input_npz = dict(np.load(args.input_path))
    npz_to_relative_npz(input_npz, args.save_path, int(args.stride_window), int(args.stride_step))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Global to Relative NPZ Converter',
                    description='Converts global Donkey car datasets into relative batches',
                    epilog='Made by Gabriel Wagner')
    parser.add_argument('input_path')
    parser.add_argument('save_path')
    parser.add_argument('stride_window')
    parser.add_argument('stride_step')
    main(parser.parse_args())