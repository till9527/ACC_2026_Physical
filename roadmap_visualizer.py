import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from custom_roadmap import CustomRoadMap
from PIL import Image

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
        print("\n[ERROR] Path generation failed! Tracing nodeSequence for errors...")

        # --- NEW DEBUGGING LOGIC ---
        for i in range(len(nodeSequence) - 1):
            from_node = nodeSequence[i]
            to_node = nodeSequence[i + 1]

            edge_found = False
            has_waypoints = False

            for edge in roadmap.edges:
                # Check if this edge connects our sequence nodes
                if (
                    roadmap.nodes.index(edge.fromNode) == from_node
                    and roadmap.nodes.index(edge.toNode) == to_node
                ):
                    edge_found = True
                    if edge.waypoints is not None:
                        has_waypoints = True
                    break

            if not edge_found:
                print(
                    f" -> FAILED: Edge {from_node} -> {to_node} does not exist in CustomRoadMap configurations!"
                )
            elif not has_waypoints:
                print(
                    f" -> FAILED: Edge {from_node} -> {to_node} exists, but curved waypoint generation failed (check coordinates/radius)."
                )
        # ---------------------------

        print("\nWe can still plot the nodes to see where the geometry is breaking!")
        plot_map(roadmap, None)
        return

    print("Path generated successfully! Launching visualizer...")
    plot_map(roadmap, waypointSequence)


def plot_map(roadmap, waypointSequence):
    """
    Plots the map image, the individual calibrated nodes, and the generated path,
    then rotates, resizes, and composites the output over the reference photo.
    """
    fig, ax = plt.subplots(figsize=(10, 12))

    # Plot the generated SCSPath (The driving line)
    if waypointSequence is not None:
        ax.plot(
            waypointSequence[0, :],
            waypointSequence[1, :],
            color="#1f77b4",
            linewidth=2.5,
            label="Generated Path",
        )

    # Plot the Node Points and their IDs
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
    ax.set_aspect("equal", adjustable="box")  # Keeps the map from stretching
    ax.axis("off")  # Hide axes for a cleaner look

    # --- GET SCRIPT DIRECTORY AND BUILD ABSOLUTE PATHS ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "ar_roadmap_overlay.png")
    background_photo_path = os.path.join(script_dir, "IMG_4009.png")
    composited_filename = os.path.join(script_dir, "composited_roadmap.png")

    # 1. Save the initial transparent plot
    plt.savefig(
        output_filename, transparent=True, bbox_inches="tight", pad_inches=0, dpi=300
    )

    # Close the matplotlib figure so it doesn't pop up and pause the script
    plt.close(fig)

    # 2. Post-process the image to rotate, resize, and OVERLAY
    try:
        print("Processing final image rotation, resizing, and compositing...")

        # Load the newly generated transparent overlay and convert to RGBA (for transparency)
        overlay = Image.open(output_filename).convert("RGBA")

        # Rotate 90 degrees (Use Image.ROTATE_90 if it rotates the wrong direction)
        overlay = overlay.transpose(Image.ROTATE_270)

        # Try to load the background photo, match sizes, and composite
        try:
            bg_img = Image.open(background_photo_path).convert("RGBA")
            target_size = bg_img.size
            print(f" -> Found background image. Target size: {target_size}")
            
            # --- NEW SCALING AND CENTERING LOGIC ---
            # Adjust this number to tweak the fit! (0.92 = 92% of the photo size)
            SCALE_FACTOR = 0.92 
            
            new_width = int(target_size[0] * SCALE_FACTOR)
            new_height = int(target_size[1] * SCALE_FACTOR)
            
            # 1. Resize the overlay to the slightly smaller size
            overlay_scaled = overlay.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 2. Create a blank, transparent canvas the exact size of the background photo
            centered_overlay = Image.new("RGBA", target_size, (0, 0, 0, 0))
            
            # 3. Calculate the math to paste the roadmap perfectly in the center
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            
            # 4. Paste the scaled roadmap onto the transparent canvas
            centered_overlay.paste(overlay_scaled, (paste_x, paste_y))
            
            # 5. Composite the perfectly centered canvas onto the background photo
            bg_img.alpha_composite(centered_overlay)
            # ---------------------------------------
            
            # Save the final merged image
            # Convert back to RGB to save as JPG (optional, but good for file size)
            bg_img.convert("RGB").save(composited_filename)
            print(f"SUCCESS: Exported fully composited image to {composited_filename}")
            
            # Optional: Save the standalone transparent overlay in case you need it
            overlay.save(output_filename)
            
        except FileNotFoundError:
            # Fallback to a standard iPhone landscape size if the photo is missing
            target_size = (4032, 3024)
            print(f" -> {background_photo_path} not found. Using default size {target_size}")
            overlay = overlay.resize(target_size, Image.Resampling.LANCZOS)
            overlay.save(output_filename)
            print(f"SUCCESS: Exported {output_filename} (Rotated and Resized)")

    except Exception as e:
        print(f"Error processing image rotation/resizing: {e}")


if __name__ == "__main__":
    main()
