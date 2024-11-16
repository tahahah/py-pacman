import os
import pygame as pg
import numpy as np
from src.env.pacman_env import PacmanEnv

def add_game_elements(layout):
    """Add player and ghost start positions to the layout if they don't exist"""
    height, width = layout.shape
    center_x = width // 2
    center_y = height // 2
    
    # Check if player start exists
    if not np.any(layout == 40):  # Player start
        print("Adding player start position...")
        # Find a suitable location for player start (near center, on empty space)
        player_placed = False
        search_radius = 0
        while not player_placed and search_radius < max(width, height):
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    y = center_y + dy
                    x = center_x + dx
                    if 0 <= y < height and 0 <= x < width:
                        if layout[y][x] in [10, 14]:  # Empty space or pellet
                            layout[y][x] = 40  # Player start
                            player_placed = True
                            print(f"Placed player at ({x}, {y})")
                            break
                if player_placed:
                    break
            search_radius += 1
    
    # Check if ghost starts exist
    ghost_count = sum(1 for i in range(33, 37) if np.any(layout == i))
    if ghost_count < 4:
        print(f"Adding {4 - ghost_count} ghost start positions...")
        # Add ghost starts (33-36) in corners or suitable locations
        ghost_positions = [
            (1, 1),                    # Top-left
            (width-2, 1),              # Top-right
            (1, height-2),             # Bottom-left
            (width-2, height-2)        # Bottom-right
        ]
        
        ghost_id = 33
        for x, y in ghost_positions:
            if ghost_id > 36:  # Only place up to 4 ghosts
                break
            if 0 <= y < height and 0 <= x < width:
                if layout[y][x] in [10, 14]:  # Empty space or pellet
                    layout[y][x] = ghost_id
                    print(f"Placed ghost {ghost_id} at ({x}, {y})")
                    ghost_id += 1
    
    return layout

def preview_layout(layout_path):
    """
    Preview a layout using the actual game renderer
    
    Args:
        layout_path: Full path to the layout file
    """
    env = None
    try:
        # First verify we can load the layout
        layout = np.loadtxt(layout_path, dtype=np.int32, skiprows=1)  # Skip the initial newline
        print(f"\nLayout loaded successfully: {layout.shape[1]}x{layout.shape[0]}")
        
        # Check and add game elements if needed
        layout = add_game_elements(layout)
        
        # Verify player position exists
        player_pos = np.where(layout == 40)
        if len(player_pos[0]) == 0:
            raise ValueError("Player position not found after adding game elements!")
        
        print(f"Player position: ({player_pos[1][0]}, {player_pos[0][0]})")
        
        # Save back if modified
        if not np.any(layout == 40) or sum(1 for i in range(33, 37) if np.any(layout == i)) < 4:
            print("Saving modified layout with added game elements...")
            with open(layout_path, 'w', newline='') as f:
                # Save without initial newline since Map.loadtxt doesn't skip it
                for row in layout:
                    line = ' '.join(str(int(x)) for x in row).rstrip()
                    f.write(line + '\n')
            
            # Verify the save worked
            test_layout = np.loadtxt(layout_path, dtype=np.int32)  # No skiprows needed
            test_player = np.where(test_layout == 40)
            if len(test_player[0]) == 0:
                raise ValueError("Player position not found in saved layout!")
            print("Layout saved and verified successfully")
        
        # Get layout name without extension and path
        layout_name = os.path.splitext(os.path.basename(layout_path))[0]
        
        # Initialize environment with just the layout name (not the full path)
        env = PacmanEnv(layout=layout_name, enable_render=True)
        
        # Reset to get initial state
        env.reset()
        
        # Render once
        env.render(mode='human')
        pg.display.flip()
        
        print(f"\nDisplaying layout '{layout_name}'")
        input("Press Enter to close the preview...")
        
    except Exception as e:
        print(f"\nError previewing layout: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
    finally:
        # Cleanup
        if env is not None:
            try:
                if hasattr(env, 'close'):
                    env.close()
            except:
                pass  # Ignore close errors
            
        if pg.get_init():
            pg.quit()

if __name__ == "__main__":
    # Get all available layouts
    layout_dir = os.path.join('res', 'layouts')
    if not os.path.exists(layout_dir):
        print(f"Error: Layout directory not found: {layout_dir}")
        exit(1)
    
    layout_files = [f for f in os.listdir(layout_dir) if f.endswith('.lay')]
    
    if not layout_files:
        print(f"No layout files found in {layout_dir}")
        exit(1)
    
    print("\nAvailable layouts:")
    for i, layout in enumerate(layout_files, 1):
        # Show layout dimensions
        try:
            layout_path = os.path.join(layout_dir, layout)
            layout_data = np.loadtxt(layout_path, dtype=np.int32, skiprows=1)  # Skip the initial newline
            size_str = f"{layout_data.shape[1]}x{layout_data.shape[0]}"
        except:
            size_str = "invalid"
        print(f"{i}. {layout} ({size_str})")
    
    while True:
        try:
            choice = input("\nEnter layout number to preview (or 'q' to quit): ")
            if choice.lower() == 'q':
                break
            
            layout_idx = int(choice) - 1
            if 0 <= layout_idx < len(layout_files):
                layout_path = os.path.join(layout_dir, layout_files[layout_idx])
                preview_layout(layout_path)
            else:
                print(f"Please enter a number between 1 and {len(layout_files)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")
