import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class FrameViewer:
    def __init__(self, hdf5_file_path, dataset_name):
        self.hdf5_file = h5py.File(hdf5_file_path, 'r')
        self.dataset = self.hdf5_file[dataset_name]
        self.num_frames = len(self.dataset)
        self.current_frame = 0

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_frame()

    def update_frame(self):
        self.ax.clear()
        frame = self.dataset[self.current_frame]
        self.ax.imshow(frame)
        self.ax.set_title(f'Frame {self.current_frame}')
        self.ax.axis('off')
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.current_frame = (self.current_frame + 1) % self.num_frames
        elif event.key == 'left':
            self.current_frame = (self.current_frame - 1) % self.num_frames
        self.update_frame()

    def show(self):
        plt.show()

if __name__ == '__main__':
    hdf5_file_path = 'pacman_data.hdf5'
    dataset_name = 'game_frame'  # Change to 'next_game_frame' to display next frames

    viewer = FrameViewer(hdf5_file_path, dataset_name)
    viewer.show()