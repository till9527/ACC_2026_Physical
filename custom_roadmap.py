import numpy as np
from hal.utilities.path_planning import RoadMap, hermite_position, SCSPath

# ==========================================
# SCALING FACTOR - single source of truth
# Node positions and non-zero radii are multiplied by this factor.
# ==========================================
SCALE_FACTOR = 1.0

# ==========================================
# TRANSLATION OFFSET - move the roadmap without changing its shape
# ==========================================
X_OFFSET = 0
Y_OFFSET = 0

# ==========================================
# DEFAULT QUANSER NODE COORDINATES
# ==========================================
NODE_DATA_BASE = {
    0: (-0.2877, 0.1998, -90.0),
    1: (0.0221, 0.1821, 90.0),
    2: (1.1552, -0.8890, 0.0),
    3: (1.1286, -0.6234, 180.0),
    4: (2.4210, 0.3946, 90.0),
    5: (2.1200, 0.3326, -90.0),
    6: (0.8011, 1.4037, 180.0),
    7: (1.3145, 1.1204, 0.0),
    8: (-1.3322, 1.2444, 180.0),
    9: (-1.3145, 0.9788, 0.0),
    10: (-1.4296, -0.4818, -42.0),
    11: (-0.3231, 2.3508, -90.0),
    12: (-0.3142, 1.9702, -90.0),
    13: (0.0044, 1.9702, 90.0),
    14: (2.3856, 3.3500, 90.0),
    15: (2.1200, 2.1384, -90.0),
    16: (0.7568, 4.0062, -80.6),
    17: (1.3322, 3.5459, -9.4),
    18: (0.3142, 3.3334, -138.0),
    19: (0.5267, 3.1298, 42.0),
    20: (-0.1726, 4.7409, 180.0),
    21: (-0.1992, 4.4310, 0.0),
    22: (-2.4741, 2.8997, -90.0),
    23: (-2.1731, 2.9439, 90.0),
}


class CustomRoadMap(RoadMap):
    """
    Hybrid RoadMap: Default node positions + proper road geometry radii.
    """

    def __init__(self, scale_factor=None, x_offset=None, y_offset=None):
        super().__init__()
        if scale_factor is None:
            scale_factor = SCALE_FACTOR
        if x_offset is None:
            x_offset = X_OFFSET
        if y_offset is None:
            y_offset = Y_OFFSET

        self.scale_factor = float(scale_factor)
        self.x_offset = float(x_offset)
        self.y_offset = float(y_offset)

        node_data = {
            node_id: (
                x * self.scale_factor + self.x_offset,
                y * self.scale_factor + self.y_offset,
                heading,
            )
            for node_id, (x, y, heading) in NODE_DATA_BASE.items()
        }

        # Edge configurations with proper default Quanser radius values
        # Format: [from_node, to_node, turn_radius]
        edgeConfigs_base = [
            [0, 2, 0.95],
            [1, 7, 0.6217],
            [1, 8, 0.8913],
            [2, 4, 0.8913],
            [3, 1, 0.6217],
            [4, 6, 0.8913],
            [5, 3, 0.6217],
            [6, 0, 0.8913],
            [6, 8, 0.0],
            [7, 5, 0.6217],
            [8, 10, 0.7122],
            [9, 0, 0.6217],
            [9, 7, 0.0],
            [10, 1, 0.72],
            [10, 2, 0.6217],
            [1, 13, 0.0],
            [4, 14, 0.0],
            [6, 13, 0.45],
            [7, 14, 0.8913],
            [8, 23, 0.6217],
            [9, 13, 0.8913],
            [11, 12, 0.0],
            [12, 0, 0.0],
            [12, 7, 0.75],
            [12, 8, 0.6217],
            [13, 19, 0.6217],
            [14, 16, 0.82],
            [14, 20, 0.75],
            [15, 5, 0.8913],
            [15, 6, 0.6217],
            [16, 17, 0.40],     # Perfectly proportional inner curve
            [16, 18, 0.45],
            [17, 15, 0.6217],
            [17, 16, 0.73],     # Perfectly proportional outer loop
            [17, 20, 0.62],
            [18, 11, 0.7631],
            [19, 17, 0.30],
            [20, 22, 1.35],
            [21, 16, 0.45],
            [22, 9, 0.8913],
            [22, 10, 0.8913],
            [23, 21, 1],
        ]
        # Scale fixed radii (keep 0 as 0)
        edgeConfigs = [
            [from_node, to_node, radius * self.scale_factor if radius > 0 else 0]
            for from_node, to_node, radius in edgeConfigs_base
        ]

        # Add nodes with calibrated positions
        sorted_ids = sorted(node_data.keys())
        for node_id in sorted_ids:
            x, y, heading_deg = node_data[node_id]
            heading_rad = np.radians(heading_deg)
            self.add_node([x, y, heading_rad])

        # Add edges with proper radius values
        # The RoadMap class will use SCSPath to generate proper curved paths
        for edgeConfig in edgeConfigs:
            self.add_edge(*edgeConfig)

        # Check for failures but DO NOT fill with straight lines
        failed_edges = []
        for edge in self.edges:
            if edge.waypoints is None:
                from_id = self.nodes.index(edge.fromNode)
                to_id = self.nodes.index(edge.toNode)
                dist = np.linalg.norm(
                    edge.toNode.pose[:2, :] - edge.fromNode.pose[:2, :]
                )
                failed_edges.append((from_id, to_id, dist))
    


if __name__ == "__main__":
    # Test the hybrid roadmap
    roadmap = CustomRoadMap()
    print(
        f"Created roadmap with {len(roadmap.nodes)} nodes and {len(roadmap.edges)} edges"
    )
