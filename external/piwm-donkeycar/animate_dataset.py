import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

"""
This tool renders an .mp4 video from an npz dataset containing keys ['timestamps'], ['state'], and ['action'] tensors.
Modify the load and save paths as needed, then run this module via it's main method.
"""

def main():
    load_path = 'my_load_path.npz'
    save_path = 'my_save_path.mp4'

    npz = dict(np.load(load_path))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # data
    t = npz['timestamps']
    n_timesteps = len(t)
    
    # Car trajectory
    car_x = npz['state'][:,0]
    car_y = npz['state'][:,1]
    car_heading = np.deg2rad(npz['state'][:,2])
    car_velocity = npz['state'][:,3]
    steering = npz['action'][:,0]
    throttle = npz['action'][:,1]
    
    # Initialize plot elements
    marker, = ax.plot([], [], 'o', color='red', markersize=12, 
                      label='Car', zorder=5)
    state_quiver = ax.quiver([], [], [], [], color='red', scale=10, 
                       width=0.008, headwidth=5, headlength=6, zorder=4)
    action_quiver = ax.quiver([], [], [], [], color='blue', scale=10, 
                       width=0.008, headwidth=5, headlength=6, zorder=3)
    trail, = ax.plot([], [], '-', color='red', alpha=0.3, linewidth=2)

    action_text = ax.text(0,1,'', ha="center")
    
    trail_x = []
    trail_y = []
    
    # Set axis properties
    ax.set_xlim(-2, 2)
    ax.set_ylim(-4, 2)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Car Trajectory Visualization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    # Animation function
    def animate(frame):
        # Current state
        x = car_x[frame]
        y = car_y[frame]
        heading = car_heading[frame]

        # Current action
        st = steering[frame]
        th = throttle[frame]
        
        # Update position marker
        marker.set_data([x], [y])
        
        # Update state vector
        vector_length = car_velocity[frame]
        u = vector_length * np.cos(heading)
        v = vector_length * np.sin(heading)
        state_quiver.set_offsets([[x, y]])
        state_quiver.set_UVC(u, v)

        # Update action vector
        u = th * np.cos(np.deg2rad(st) + heading)
        v = th * np.sin(np.deg2rad(st) + heading)

        action_quiver.set_offsets([[x,y]])
        action_quiver.set_UVC(u, v)
        
        # Update trajectory trail
        trail_x.append(x)
        trail_y.append(y)
        trail.set_data(trail_x, trail_y)
        
        ax.set_title(f'Car Trajectory Visualization', 
                     fontsize=14, fontweight='bold')

        action_text.set_text(f"Frame: {frame}\nSteering Action: {st}\nThrottle Action: {th}")
        
        return marker, state_quiver, action_quiver, trail, action_text,
    
    # Create animation
    start_frame = 0

    anim = FuncAnimation(fig, animate, frames=range(start_frame, n_timesteps), 
                         interval=1000//24, blit=True, repeat=True)
    
    FFwriter = FFMpegWriter(fps=24)
    update_func = lambda _i, _n: progress_bar.update(1)
    with tqdm(total=n_timesteps, desc='Saving Video') as progress_bar:
        anim.save(save_path, writer=FFwriter, dpi=100, progress_callback=update_func)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()