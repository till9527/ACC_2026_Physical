import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from custom_roadmap import CustomRoadMap

# 1. Define the exact node sequence from vehicle_control.py
nodeSequence = [
    # Lap 1
    10,
    2,
    4,
    6,
    8,
    23,
    21,
    16,
    18,
    11,
    12,
    7,
    14,
    20,
    22,
    9,
    13,
    19,
    17,
    15,
    5,
    3,
    1,
    8,
    # Lap 2
    10,
    1,
    7,
    5,
    3,
    1,
    13,
    19,
    17,
    20,
    22,
    # Lap 3
    10,
    2,
    4,
    14,
    16,
    17,
    15,
    6,
    0,
    2,
    4,
    6,
    13,
    19,
    17,
    16,
    18,
    11,
    12,
    8,
    # Lap 4
    10,
    1,
    8,
    23,
    21,
    16,
    18,
    11,
    12,
    0,
    2,
    4,
    14,
    20,
    22,
    9,
    7,
    14,
    20,
    22,
    9,
    0,
    2,
    4,
    6,
    8,
    10,
]


def main():
    print("Initializing CustomRoadMap...")
    roadmap = CustomRoadMap()

    print("Generating Path Waypoints...")
    waypointSequence = roadmap.generate_path(nodeSequence)

    if waypointSequence is None:
        print("\n[ERROR] Path generation returned None. A curve failed to generate.")
        print("Check the terminal output of CustomRoadMap for failing edges.")
        # We can still plot the nodes to see where the geometry is breaking!
        plot_map(roadmap, None)
        return

    print("Path generated successfully! Launching visualizer...")
    plot_map(roadmap, waypointSequence)


def plot_map(roadmap, waypointSequence):
    """
    Plots the map image, the individual calibrated nodes, and the generated path.
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_title("Offline Custom Roadmap Visualizer", fontsize=16)

    # 2. Load and display the background map image

    # 3. Plot the generated SCSPath (The driving line)
    if waypointSequence is not None:
        ax.plot(
            waypointSequence[0, :],
            waypointSequence[1, :],
            color="#1f77b4",
            linewidth=2.5,
            label="Generated Path",
        )

    # 4. Plot the Node Points and their IDs for easy debugging
    node_x = []
    node_y = []

    for i, node in enumerate(roadmap.nodes):
        x = node.pose[0, 0]
        y = node.pose[1, 0]
        node_x.append(x)
        node_y.append(y)

        # Annotate each dot with its Node ID
        ax.annotate(
            str(i),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="darkred",
        )

    ax.scatter(node_x, node_y, color="red", s=50, zorder=5, label="Calibrated Nodes")

    # Formatting
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-1.5, 5.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")  # Keeps the map from stretching

    plt.show()


if __name__ == "__main__":
    main()
