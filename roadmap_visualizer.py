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
        self.fig.canvas.manager.set_window_title('Interactive Dynamic Curve Tuner')
        
        self.selected_node_id = None
        self.scatter_plot = None
        self.path_line = None
        self.node_texts = {}
        
        self.setup_plot()
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def check_map_health(self, roadmap):
        """Scans every edge in the roadmap and reports any failures to the terminal."""
        print("\n=== EDGE HEALTH REPORT ===")
        failed_count = 0
        
        for edge in roadmap.edges:
            # If waypoints is None, or it's an empty array, the curve failed
            if getattr(edge, 'waypoints', None) is None or len(edge.waypoints) == 0 or len(edge.waypoints[0]) == 0:
                from_id = roadmap.nodes.index(edge.fromNode)
                to_id = roadmap.nodes.index(edge.toNode)
                print(f" [FAILED] Edge {from_id} -> {to_id} could not generate geometry!")
                failed_count += 1
                
        if failed_count == 0:
            print(" [OK] All edges generated perfectly!")
        else:
            print(f" >>> WARNING: {failed_count} edges are broken on the map! <<<")
        print("==========================\n")

    def setup_plot(self):
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.axis("off")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, "IMG_4009.png")
        try:
            bg_img = Image.open(img_path).transpose(Image.ROTATE_90)
            self.ax.imshow(bg_img, extent=[-3.0, 3.0, -1.5, 5.5], alpha=0.7)
        except FileNotFoundError:
            pass

        # Generate initial path
        roadmap = CustomRoadMap()
        
        # Run our diagnostic check immediately on startup!
        self.check_map_health(roadmap)
        
        waypointSequence = roadmap.generate_path(nodeSequence)

        if waypointSequence is not None:
            (self.path_line,) = self.ax.plot(
                waypointSequence[0, :], waypointSequence[1, :],
                color="#1f77b4", linewidth=2.5, zorder=2
            )
        else:
            (self.path_line,) = self.ax.plot([], [], color="#1f77b4", linewidth=2.5, zorder=2)

        # Plot Nodes
        node_x = [node.pose[0, 0] for node in roadmap.nodes]
        node_y = [node.pose[1, 0] for node in roadmap.nodes]
        
        self.scatter_plot = self.ax.scatter(node_x, node_y, color="red", s=80, zorder=5, picker=True)

        for i, node in enumerate(roadmap.nodes):
            txt = self.ax.annotate(
                str(i), (node_x[i], node_y[i]),
                textcoords="offset points", xytext=(5, 5),
                ha="center", fontsize=11, fontweight="bold", color="darkred", zorder=6
            )
            self.node_texts[i] = txt

    def on_press(self, event):
        if event.inaxes != self.ax: return
        min_dist = float('inf')
        closest_node = None
        for node_id, coords in custom_roadmap.NODE_DATA_BASE.items():
            dist = math.hypot(event.xdata - coords[0], event.ydata - coords[1])
            if dist < min_dist:
                min_dist = dist
                closest_node = node_id
        if min_dist < 0.2:
            self.selected_node_id = closest_node

    def on_motion(self, event):
        if self.selected_node_id is None or event.inaxes != self.ax: return
        
        # Snap visuals to mouse pointer
        offsets = self.scatter_plot.get_offsets()
        keys_list = list(sorted(custom_roadmap.NODE_DATA_BASE.keys()))
        idx = keys_list.index(self.selected_node_id)
        
        offsets[idx] = [event.xdata, event.ydata]
        self.scatter_plot.set_offsets(offsets)
        self.node_texts[self.selected_node_id].set_position((event.xdata, event.ydata))
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.selected_node_id is None or event.xdata is None: return
        
        # 1. Update the coordinate dictionary in memory
        old_val = custom_roadmap.NODE_DATA_BASE[self.selected_node_id]
        custom_roadmap.NODE_DATA_BASE[self.selected_node_id] = (round(event.xdata, 4), round(event.ydata, 4), old_val[2])
        
        # 2. Re-instantiate the CustomRoadMap to apply new curves
        new_roadmap = CustomRoadMap()
        
        # Run our diagnostic check every time you drop a node!
        self.check_map_health(new_roadmap)
        
        waypointSequence = new_roadmap.generate_path(nodeSequence)
        
        # 3. Update the blue line
        if waypointSequence is not None:
            self.path_line.set_data(waypointSequence[0, :], waypointSequence[1, :])
        else:
            print(f"[ERROR] Node sequence broke! The required path cannot be completed.")
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
    app = InteractiveRoadmapTool()
    plt.show()