import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mazelib import Maze
from mazelib.generate.Prims import Prims
import random
import time

class PacmanLayoutGenerator:
    def __init__(self):
        self.output_dir = os.path.join('res', 'layouts')
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.current_layout = None
        self.setup_buttons()
        
        # Colors for visualization
        self.COLORS = {
            'wall': '#0000FF',        # Blue for walls (16-33)
            'empty': '#000000',       # Black for empty space (10)
            'pellet': '#FFB897',      # Light orange for pellets (14)
            'power_pellet': '#FFF400', # Yellow for power pellets (15)
            'pacman': '#FFFF00',      # Yellow for Pacman (40)
            'ghost': '#FF0000',       # Red for ghosts (33-36)
        }

    def setup_buttons(self):
        # Create save and discard buttons
        self.save_button_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.discard_button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        
        self.save_button = Button(self.save_button_ax, 'Save', color='lightgreen')
        self.discard_button = Button(self.discard_button_ax, 'Discard', color='lightcoral')
        
        self.save_button.on_clicked(self.save_layout)
        self.discard_button.on_clicked(self.generate_new)

    def generate_maze(self):
        # Random dimensions (odd numbers for maze generation)
        height = random.randrange(11, 22, 2)
        width = random.randrange(11, 22, 2)
        
        # Generate base maze
        m = Maze()
        m.generator = Prims(height, width)
        m.generate()
        
        # Convert maze to our format (16-33 for walls, 10 for paths)
        maze = m.grid.astype(int)
        maze[maze == 1] = 16  # Convert walls
        maze[maze == 0] = 10  # Convert paths
        
        return maze

    def add_game_elements(self, maze):
        # Convert to our format
        layout = maze.copy()
        empty_positions = np.argwhere(layout == 10)
        
        # Add pellets to empty spaces
        for pos in empty_positions:
            if random.random() < 0.7:  # 70% chance for pellets
                layout[pos[0], pos[1]] = 14
        
        # Add power pellets (in corners or strategic positions)
        power_pellet_count = 4
        power_pellet_positions = []
        while len(power_pellet_positions) < power_pellet_count:
            pos = empty_positions[random.randint(0, len(empty_positions) - 1)]
            if layout[pos[0], pos[1]] in [10, 14]:  # Only place on empty or pellet spaces
                layout[pos[0], pos[1]] = 15
                power_pellet_positions.append(pos)
        
        # Add Pacman
        pacman_placed = False
        while not pacman_placed:
            pos = empty_positions[random.randint(0, len(empty_positions) - 1)]
            if layout[pos[0], pos[1]] in [10, 14]:  # Place on empty or pellet space
                layout[pos[0], pos[1]] = 40
                pacman_placed = True
        
        # Add ghosts (33-36)
        ghost_count = 4
        ghost_positions = []
        ghost_id = 33
        while len(ghost_positions) < ghost_count and ghost_id <= 36:
            pos = empty_positions[random.randint(0, len(empty_positions) - 1)]
            if layout[pos[0], pos[1]] in [10, 14]:  # Place on empty or pellet space
                layout[pos[0], pos[1]] = ghost_id
                ghost_positions.append(pos)
                ghost_id += 1
        
        return layout

    def display_layout(self, layout):
        self.ax.clear()
        
        # Create color map
        cmap = plt.cm.colors.ListedColormap([
            self.COLORS['empty'],      # 10: empty
            self.COLORS['pellet'],     # 14: pellet
            self.COLORS['power_pellet'], # 15: power pellet
            self.COLORS['wall'],       # 16: wall
            self.COLORS['ghost'],      # 33-36: ghosts
            self.COLORS['pacman'],     # 40: pacman
        ])
        
        # Display the layout
        self.ax.imshow(layout, cmap=cmap)
        
        # Add grid
        self.ax.grid(True, which='both', color='gray', linewidth=0.5)
        
        # Add numbers to cells
        for i in range(layout.shape[0]):
            for j in range(layout.shape[1]):
                self.ax.text(j, i, str(layout[i, j]), 
                           ha='center', va='center', 
                           color='white', fontsize=8)
        
        plt.tight_layout()
        self.fig.canvas.draw_idle()

    def save_layout(self, event):
        if self.current_layout is not None:
            timestamp = int(time.time())
            filename = os.path.join(self.output_dir, f'generated_maze_{timestamp}.lay')
            np.savetxt(filename, self.current_layout, fmt='%d')
            print(f"Saved layout to: {filename}")
        self.generate_new(event)

    def generate_new(self, event):
        maze = self.generate_maze()
        self.current_layout = self.add_game_elements(maze)
        self.display_layout(self.current_layout)

    def run(self):
        self.generate_new(None)  # Generate first layout
        plt.show()

if __name__ == '__main__':
    generator = PacmanLayoutGenerator()
    generator.run()
