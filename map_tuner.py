import time
import cv2
import numpy as np
import pyqtgraph as pg
import os
import contextlib

# --- Quanser GUI & Math Imports ---
from pal.utilities.scope import MultiScope
import pal.resources.images as images

# --- Quanser Standard Map Import ---
from hal.products.mats import SDCSRoadMap


def main():
    print("Loading Quanser SDCSRoadMap...")
    # Initialized exactly like in vehicle_control.py
    roadmap = SDCSRoadMap(leftHandTraffic=False)

    # 1. Brute-force discover ALL nodes in the file so we can label them
    all_nodes = []
    node_x = []
    node_y = []

    print("Scanning for valid nodes...")
    # Scan IDs 0 to 40 to find valid nodes (Standard map usually has around 24)
    for i in range(40):
        try:
            pose = roadmap.get_node_pose(i)
            if pose is not None:
                pose = pose.squeeze()
                all_nodes.append(i)
                node_x.append(pose[0])
                node_y.append(pose[1])
        except Exception:
            pass

    # =========================================================
    # UI SETUP
    # =========================================================
    scope = MultiScope(rows=1, cols=1, title="Offline Path Tuner - Quanser Map", fps=30)

    scope.addXYAxis(
        row=0,
        col=0,
        xLabel="x Position [m]",
        yLabel="y Position [m]",
        xLim=(-2.5, 2.5),
        yLim=(-1, 5),
    )

    # Attach the Quanser Cityscape Background Image
    im = cv2.imread(images.SDCS_CITYSCAPE, cv2.IMREAD_GRAYSCALE)
    scope.axes[0].attachImage(
        scale=(-0.002035, 0.002035), offset=(1125, 2365), rotation=180, levels=(0, 255)
    )
    scope.axes[0].images[0].setImage(image=im)

    # 2. Brute-force every possible edge and draw valid ones
    print(
        f"Discovered {len(all_nodes)} nodes. Brute-forcing all possible connections..."
    )
    print("This may take a few seconds...")

    for start_node in all_nodes:
        for end_node in all_nodes:
            if start_node == end_node:
                continue

            # Temporarily suppress stdout to hide Quanser error spam for invalid paths
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                try:
                    path = roadmap.generate_path([start_node, end_node])
                    valid = True
                except Exception:
                    valid = False

            if valid and path is not None and len(path) > 0:
                segment = pg.PlotDataItem(
                    pen={"color": (85, 168, 104), "width": 2},
                    name=f"Path_{start_node}_{end_node}",
                )
                scope.axes[0].plot.addItem(segment)
                segment.setData(path[0, :], path[1, :])

    # 3. Draw ALL available nodes (Red Dots)
    print("Drawing reference nodes...")
    nodeScatter = pg.ScatterPlotItem(
        size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 200)
    )
    nodeScatter.setData(node_x, node_y)
    scope.axes[0].plot.addItem(nodeScatter)

    # 4. Add Text Labels for every node ID
    for i, node in enumerate(all_nodes):
        text = pg.TextItem(f"{node}", color=(255, 0, 0), anchor=(0.5, 1.5))
        text.setPos(node_x[i], node_y[i])

        font = text.textItem.font()
        font.setPointSize(10)
        font.setBold(True)
        text.textItem.setFont(font)

        scope.axes[0].plot.addItem(text)

    print("\nMap loaded successfully! Leave this window open to view.")

    # Keep the window open and refreshing
    try:
        while True:
            MultiScope.refreshAll()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nExiting Map Tuner...")


if __name__ == "__main__":
    main()
