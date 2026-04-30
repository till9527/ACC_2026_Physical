import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import custom_roadmap  # Import the module so we can modify its dictionary in memory
from custom_roadmap import CustomRoadMap

# 1. Define the exact node sequence
nodeSequence = [
    # Lap 1
    10, 2, 4, 6, 8, 23, 21, 16, 18, 11, 12, 7, 14, 20, 22, 9, 13, 19, 17, 15, 5, 3, 1, 8,
    # Lap 2
    10, 1, 7, 5, 3, 1, 13, 19, 17, 20, 22,
    # Lap 3
    10, 2, 4, 14, 16, 17, 15, 6, 0, 2, 4, 6, 13, 19, 17, 16, 18, 11, 12, 8,
    # Lap 4
    10, 1, 8, 23, 21, 16, 18, 11, 12, 0, 2, 4, 14, 20, 22, 9, 7, 14, 20, 22, 9, 0, 2, 4, 6, 8, 10,
]

class InteractiveRoadmapTool:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 12))
        self.fig.canvas.manager.set_window_title('Interactive Roadmap Tuner')
        
        self.selected_node_id = None
        self.scatter_plot = None
        self.path_line = None
        self.node_texts = {}
        
        # Setup the initial plot
        self.setup_plot()
        
        # Connect mouse events for dragging
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def setup_plot(self):
        """Initializes the background image, nodes, and lines."""
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.axis("off")
        
        # --- 1. Load Background Image ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, "IMG_4009.png")
        try:
            # We rotate it 90 degrees so it stands upright like your roadmap
            bg_img = Image.open(img_path).transpose(Image.ROTATE_90)
            
            # EXTENT defines where the image sits in the math coordinates [left, right, bottom, top]
            # You can tweak these 4 numbers to shift/scale the photo under your nodes!
            self.ax.imshow(bg_img, extent=[-3.0, 3.0, -1.5, 5.5], alpha=0.7)
        except FileNotFoundError:
            print(f"[WARNING] Could not find {img_path} to use as background.")

        # --- 2. Generate Roadmap ---
        roadmap = CustomRoadMap()
        waypointSequence = roadmap.generate_path(nodeSequence)

        # --- 3. Plot Paths ---
        if waypointSequence is not None:
            (self.path_line,) = self.ax.plot(
                waypointSequence[0, :], waypointSequence[1, :],
                color="#1f77b4", linewidth=2.5, zorder=2
            )
        else:
            print("[WARNING] Path generation failed. Curves may be impossible with current coordinates.")
            (self.path_line,) = self.ax.plot([], [], color="#1f77b4", linewidth=2.5, zorder=2)

        # --- 4. Plot Nodes ---
        node_x = [node.pose[0, 0] for node in roadmap.nodes]
        node_y = [node.pose[1, 0] for node in roadmap.nodes]
        
        self.scatter_plot = self.ax.scatter(node_x, node_y, color="red", s=80, zorder=5, picker=True)

        self.node_texts = {}
        for i, node in enumerate(roadmap.nodes):
            txt = self.ax.annotate(
                str(i), (node_x[i], node_y[i]),
                textcoords="offset points", xytext=(5, 5),
                ha="center", fontsize=11, fontweight="bold", color="darkred", zorder=6
            )
            self.node_texts[i] = txt

    def on_press(self, event):
        """Fires when the mouse is clicked."""
        if event.inaxes != self.ax: return
        
        # Find the node closest to the click
        min_dist = float('inf')
        closest_node = None
        
        for node_id, coords in custom_roadmap.NODE_DATA_BASE.items():
            dist = math.hypot(event.xdata - coords[0], event.ydata - coords[1])
            if dist < min_dist:
                min_dist = dist
                closest_node = node_id
                
        # If click is within a small radius (0.2 meters), select it
        if min_dist < 0.2:
            self.selected_node_id = closest_node
            print(f"-> Picked up Node {self.selected_node_id}")

    def on_motion(self, event):
        """Fires when the mouse is moved."""
        if self.selected_node_id is None or event.inaxes != self.ax: return
        
        # Update just the visual dot and text (don't recalculate heavy paths yet)
        offsets = self.scatter_plot.get_offsets()
        offsets[self.selected_node_id] = [event.xdata, event.ydata]
        self.scatter_plot.set_offsets(offsets)
        
        self.node_texts[self.selected_node_id].set_position((event.xdata, event.ydata))
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        """Fires when the mouse is released. Recalculates paths."""
        if self.selected_node_id is None or event.xdata is None: return
        
        print(f"-> Dropped Node {self.selected_node_id} at X: {event.xdata:.4f}, Y: {event.ydata:.4f}")
        
        # 1. Update the coordinate dictionary in memory
        old_val = custom_roadmap.NODE_DATA_BASE[self.selected_node_id]
        custom_roadmap.NODE_DATA_BASE[self.selected_node_id] = (round(event.xdata, 4), round(event.ydata, 4), old_val[2])
        
        # 2. Re-instantiate the CustomRoadMap to apply new curves
        new_roadmap = CustomRoadMap()
        waypointSequence = new_roadmap.generate_path(nodeSequence)
        
        # 3. Update the blue line
        if waypointSequence is not None:
            self.path_line.set_data(waypointSequence[0, :], waypointSequence[1, :])
        else:
            print(f"[ERROR] Moving node {self.selected_node_id} broke the curve geometry!")
            self.path_line.set_data([], [])

        self.fig.canvas.draw_idle()
        self.selected_node_id = None
        
        # 4. Print the updated dictionary to the terminal for copy-pasting
        print("\n--- CURRENT NODE DICTIONARY ---")
        for k, v in custom_roadmap.NODE_DATA_BASE.items():
            print(f"    {k}: ({v[0]:.4f}, {v[1]:.4f}, {v[2]}),")
        print("-------------------------------\n")

if __name__ == "__main__":
    print("Launching Interactive Roadmap Tuner...")
    print("Instructions:")
    print(" - Click and hold a red node to drag it.")
    print(" - Release to snap it into place and recalculate curves.")
    print(" - The terminal will output your new coordinates.")
    
    app = InteractiveRoadmapTool()
    plt.show()