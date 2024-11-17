import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
import argparse

# Define colors for different elements
COLORS = {
    'wall': '#0000FF',        # Blue for walls (16-33)
    'empty': '#000000',       # Black for empty space (10)
    'pellet': '#FFB897',      # Light orange for pellets (14)
    'power_pellet': '#FFF400', # Yellow for power pellets (15)
    'pacman': '#FFFF00',      # Yellow for Pacman (40)
    'ghost': '#FF0000',       # Red for ghosts (33-36)
    'unused': '#000000',      # Black for unused space (50)
}

def create_color_map(layout):
    # Create a colormap based on the tile values
    unique_values = np.unique(layout)
    colors = []
    for value in unique_values:
        if 16 <= value <= 33:
            colors.append(COLORS['wall'])
        elif value == 10:
            colors.append(COLORS['empty'])
        elif value == 14:
            colors.append(COLORS['pellet'])
        elif value == 15:
            colors.append(COLORS['power_pellet'])
        elif value == 40:
            colors.append(COLORS['pacman'])
        elif 33 <= value <= 36:
            colors.append(COLORS['ghost'])
        elif value == 50:
            colors.append(COLORS['unused'])
        else:
            colors.append('#FFFFFF')  # White for unknown values
    
    return ListedColormap(colors)

def visualize_layout(layout_path, save=False):
    # Load the layout
    layout = np.loadtxt(layout_path).astype(int)
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Create custom colormap
    cmap = create_color_map(layout)
    
    # Plot the layout
    plt.imshow(layout, cmap=cmap)
    
    # Add grid
    plt.grid(True, which='both', color='gray', linewidth=0.5)
    
    # Add title
    layout_name = os.path.basename(layout_path).replace('.lay', '')
    plt.title(f'Pacman Layout: {layout_name}')
    
    # Add numbers to cells
    for i in range(layout.shape[0]):
        for j in range(layout.shape[1]):
            plt.text(j, i, str(layout[i, j]), 
                    ha='center', va='center', 
                    color='white', fontsize=8)
    
    plt.tight_layout()
    
    if save:
        output_path = f'{layout_name}_visualization.png'
        plt.savefig(output_path)
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

def visualize_all_layouts(layout_dir='res/layouts', save=False):
    # Get all layout files
    layout_files = glob.glob(os.path.join(layout_dir, '*.lay'))
    
    if not layout_files:
        print(f"No layout files found in directory: {layout_dir}")
        return
        
    print(f"Found {len(layout_files)} layout files")
    for layout_file in layout_files:
        print(f"Visualizing: {layout_file}")
        visualize_layout(layout_file, save)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Pacman layout files')
    parser.add_argument('--path', type=str, help='Path to a specific layout file or directory containing layouts')
    parser.add_argument('--save', action='store_true', help='Save visualizations instead of displaying them')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    if args.path:
        if os.path.isfile(args.path):
            # Visualize single layout
            print(f"Visualizing layout: {args.path}")
            visualize_layout(args.path, args.save)
        elif os.path.isdir(args.path):
            # Visualize all layouts in directory
            print(f"Visualizing all layouts in: {args.path}")
            visualize_all_layouts(args.path, args.save)
        else:
            print(f"Error: Path not found: {args.path}")
    else:
        # Default behavior: visualize all layouts in res/layouts
        default_dir = os.path.join('res', 'layouts')
        print(f"No path specified. Using default directory: {default_dir}")
        visualize_all_layouts(default_dir, args.save)
