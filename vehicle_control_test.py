# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# region : File Description and Imports

"""
vehicle_control.py

Skills acivity code for vehicle control lab guide.
Students will implement a vehicle speed and steering controller.
Please review Lab Guide - vehicle control PDF
"""

import os
import signal
import numpy as np
from threading import Thread
import time
import cv2
import pyqtgraph as pg

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images
from custom_roadmap import CustomRoadMap

# ================ Experiment Configuration ================
# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500
tf = 6000
startDelay = 1
controllerUpdateRate = 100

# ===== Speed Controller Parameters
# - v_ref: desired velocity in m/s (OVERRIDDEN BY CSV)
# - K_p: proportional gain for speed controller
# - K_i: integral gain for speed controller
v_ref = 0.5 
K_p = 0.1
K_i = 1

# ===== Steering Controller Parameters
enableSteeringControl = True
K_stanley = 1
nodeSequence = [10, 1, 8, 10]

# --- CSV FILE MAPPING ---
# Update these filenames to match your exact saved files
CSV_FILES = {
    (10, 1): 'edge_10_1.csv',
    (1, 8):  'edge_1_8.csv',
    (8, 10): 'edge_8_10.csv'
}
# ------------------------

# endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
def check_map_health(roadmap):
    """Scans every edge in the roadmap and reports any failures to the terminal."""
    print("\n=== EDGE HEALTH REPORT ===")
    failed_count = 0

    for edge in roadmap.edges:
        if (
            getattr(edge, "waypoints", None) is None
            or len(edge.waypoints) == 0
            or len(edge.waypoints[0]) == 0
        ):
            from_id = roadmap.nodes.index(edge.fromNode)
            to_id = roadmap.nodes.index(edge.toNode)
            print(f" [FAILED] Edge {from_id} -> {to_id} could not generate geometry!")
            failed_count += 1

    if failed_count == 0:
        print(" [OK] All edges generated perfectly!")
    else:
        print(f" >>> WARNING: {failed_count} edges are broken on the map! <<<")
    print("==========================\n")


# region : Initial setup
if enableSteeringControl:
    roadmap = CustomRoadMap()
    check_map_health(roadmap)
    waypointSequence = roadmap.generate_path(nodeSequence)

    if waypointSequence is None:
        print("\n[ERROR] Path generation failed! Tracing nodeSequence for errors...")
        for i in range(len(nodeSequence) - 1):
            from_node = nodeSequence[i]
            to_node = nodeSequence[i + 1]

            edge_found = False
            has_waypoints = False

            for edge in roadmap.edges:
                if (
                    roadmap.nodes.index(edge.fromNode) == from_node
                    and roadmap.nodes.index(edge.toNode) == to_node
                ):
                    edge_found = True
                    if edge.waypoints is not None:
                        has_waypoints = True
                    break

            if not edge_found:
                print(f" -> FAILED: Edge {from_node} -> {to_node} does not exist in CustomRoadMap configurations!")
            elif not has_waypoints:
                print(f" -> FAILED: Edge {from_node} -> {to_node} exists, but curved waypoint generation failed (check coordinates/radius).")

        import sys
        sys.exit(1)

    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
else:
    initialPose = [0, 0, 0]

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]],
    )
    calibrate = False
else:
    calibrate = "y" in input("do you want to recalibrate?(y/n)")

calibrationPose = [0, 2, -np.pi / 2]

global KILL_THREAD
KILL_THREAD = False

def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True

signal.signal(signal.SIGINT, sig_handler)
# endregion


class SpeedController:
    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref_target, dt):
        e = v_ref_target - v
        self.ei += dt * e
        return np.clip(
            self.kp * e + self.ki * self.ei, -self.maxThrottle, self.maxThrottle
        )


# ==============  SECTION: Replay Manager  ====================
class ReplayManager:
    """Loads CSVs and interpolates data based on the current edge and elapsed time."""
    def __init__(self, node_sequence, file_mapping, roadmap_obj):
        self.node_sequence = node_sequence
        self.edge_data = []
        self.node_poses = [roadmap_obj.get_node_pose(n)[:2].flatten() for n in node_sequence]
        self.current_edge_idx = 0
        self.edge_start_time = 0.0
        
        # Load the CSV data for each edge sequentially
        for i in range(len(node_sequence) - 1):
            edge = (node_sequence[i], node_sequence[i+1])
            filename = file_mapping.get(edge)
            if filename and os.path.exists(filename):
                try:
                    # Expects headers: Time_s, Steering_rad, Speed_mps
                    data = np.loadtxt(filename, delimiter=',', skiprows=1)
                    # Normalize time so every edge playback starts at T=0
                    data[:, 0] -= data[0, 0] 
                    self.edge_data.append(data)
                    print(f"Loaded {filename} successfully for edge {edge}")
                except Exception as e:
                    print(f"[ERROR] Failed to read {filename}: {e}")
                    self.edge_data.append(None)
            else:
                print(f"[WARNING] Could not find file {filename} for edge {edge}")
                self.edge_data.append(None)

    def get_commands(self, t, p):
        # 1. Edge Transition Logic: Check if we reached the target node
        if self.current_edge_idx < len(self.node_poses) - 1:
            target_xy = self.node_poses[self.current_edge_idx + 1]
            dist = np.linalg.norm(p - target_xy)
            
            # If within 40cm of the node, transition to next edge data
            if dist < 0.04:
                self.current_edge_idx += 1
                self.edge_start_time = t
                if self.current_edge_idx < len(self.node_sequence) - 1:
                    print(f"Reached node! Transitioning to edge: {self.node_sequence[self.current_edge_idx]} -> {self.node_sequence[self.current_edge_idx+1]}")
                else:
                    print("Reached final node! Route Complete.")

        # 2. Check if route is finished
        if self.current_edge_idx >= len(self.edge_data) or self.edge_data[self.current_edge_idx] is None:
            return 0.0, 0.0, True # Return 0 speed, 0 steering, is_finished=True

        # 3. Interpolate commands based on time spent on THIS edge
        data = self.edge_data[self.current_edge_idx]
        t_edge = t - self.edge_start_time
        
        times = data[:, 0]
        steerings = data[:, 1]
        speeds = data[:, 2]

        # np.interp automatically holds the last value if t_edge exceeds the max recorded time
        csv_speed = np.interp(t_edge, times, speeds)
        csv_steering = np.interp(t_edge, times, steerings)

        return csv_speed, csv_steering, False


def controlLoop():
    # region controlLoop setup
    global KILL_THREAD
    u = 0
    delta = 0
    countMax = controllerUpdateRate / 10
    count = 0
    # endregion

    # region Controller initialization
    speedController = SpeedController(kp=K_p, ki=K_i)
    if enableSteeringControl:
        # Initialize Replay Manager instead of standard Steering Controller
        replayManager = ReplayManager(nodeSequence, CSV_FILES, roadmap)
    # endregion

    # region QCar interface setup
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)
    else:
        gps = memoryview(b"")
    # endregion

    with qcar, gps:
        t0 = time.time()
        t = 0
        while (t < tf + startDelay) and (not KILL_THREAD):
            # region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t - tp
            # endregion

            # region : Read from sensors and update state estimates
            qcar.read()
            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                    ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
                else:
                    ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])

                x = ekf.x_hat[0, 0]
                y = ekf.x_hat[1, 0]
                th = ekf.x_hat[2, 0]
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            else:
                p = np.array([0, 0])
            v = qcar.motorTach
            # endregion

            # region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0
                if enableSteeringControl:
                    replayManager.edge_start_time = t # Keep resetting start time until delay finishes
            else:
                if enableSteeringControl:
                    # Get replicated data from the manager based on current pose and time
                    csv_v_ref, csv_delta, route_finished = replayManager.get_commands(t, p)
                    
                    if route_finished:
                        u = speedController.update(v, 0.0, dt) # Stop car
                        delta = 0.0
                    else:
                        u = speedController.update(v, csv_v_ref, dt) # Track recorded speed
                        delta = csv_delta                            # Override steering directly
                        
                    current_v_ref = csv_v_ref if not route_finished else 0.0
                else:
                    u = speedController.update(v, v_ref, dt)
                    delta = 0
                    current_v_ref = v_ref

            qcar.write(u, delta)
            # endregion

            # region : Update Scopes
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay

                # Speed control scope
                speedScope.axes[0].sample(t_plot, [v, current_v_ref])
                speedScope.axes[1].sample(t_plot, [current_v_ref - v])
                speedScope.axes[2].sample(t_plot, [u])

                # Steering control scope
                if enableSteeringControl:
                    steeringScope.axes[4].sample(t_plot, [[p[0], p[1]]])

                    p[0] = ekf.x_hat[0, 0]
                    p[1] = ekf.x_hat[1, 0]

                    # Since we bypassed Stanley, we'll just plot current states against GPS
                    x_ref = gps.position[0]
                    y_ref = gps.position[1]
                    th_ref = gps.orientation[2]

                    steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                    steeringScope.axes[1].sample(t_plot, [p[1], y_ref])
                    steeringScope.axes[2].sample(t_plot, [th, th_ref])
                    steeringScope.axes[3].sample(t_plot, [delta])

                    arrow.setPos(p[0], p[1])
                    arrow.setStyle(angle=180 - th * 180 / np.pi)

                count = 0
            # endregion
            continue
        qcar.read_write_std(throttle=0, steering=0)


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# region : Setup and run experiment
if __name__ == "__main__":

    # region : Setup scopes
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30
    
    speedScope = MultiScope(rows=3, cols=1, title="Vehicle Speed Control", fps=fps)
    speedScope.addAxis(row=0, col=0, timeWindow=tf, yLabel="Vehicle Speed [m/s]", yLim=(0, 1))
    speedScope.axes[0].attachSignal(name="v_meas", width=2)
    speedScope.axes[0].attachSignal(name="v_ref")

    speedScope.addAxis(row=1, col=0, timeWindow=tf, yLabel="Speed Error [m/s]", yLim=(-0.5, 0.5))
    speedScope.axes[1].attachSignal()

    speedScope.addAxis(row=2, col=0, timeWindow=tf, xLabel="Time [s]", yLabel="Throttle Command [%]", yLim=(-0.3, 0.3))
    speedScope.axes[2].attachSignal()

    if enableSteeringControl:
        steeringScope = MultiScope(rows=4, cols=2, title="Vehicle Steering Control", fps=fps)

        steeringScope.addAxis(row=0, col=0, timeWindow=tf, yLabel="x Position [m]", yLim=(-2.5, 2.5))
        steeringScope.axes[0].attachSignal(name="x_meas")
        steeringScope.axes[0].attachSignal(name="x_ref")

        steeringScope.addAxis(row=1, col=0, timeWindow=tf, yLabel="y Position [m]", yLim=(-1, 5))
        steeringScope.axes[1].attachSignal(name="y_meas")
        steeringScope.axes[1].attachSignal(name="y_ref")

        steeringScope.addAxis(row=2, col=0, timeWindow=tf, yLabel="Heading Angle [rad]", yLim=(-3.5, 3.5))
        steeringScope.axes[2].attachSignal(name="th_meas")
        steeringScope.axes[2].attachSignal(name="th_ref")

        steeringScope.addAxis(row=3, col=0, timeWindow=tf, yLabel="Steering Angle [rad]", yLim=(-0.6, 0.6))
        steeringScope.axes[3].attachSignal()
        steeringScope.axes[3].xLabel = "Time [s]"

        steeringScope.addXYAxis(row=0, col=1, rowSpan=4, xLabel="x Position [m]", yLabel="y Position [m]", xLim=(-2.5, 2.5), yLim=(-1, 5))

        im = cv2.imread(images.SDCS_CITYSCAPE, cv2.IMREAD_GRAYSCALE)

        steeringScope.axes[4].attachImage(scale=(-0.002035, 0.002035), offset=(1125, 2365), rotation=180, levels=(0, 255))
        steeringScope.axes[4].images[0].setImage(image=im)

        referencePath = pg.PlotDataItem(pen={"color": (85, 168, 104), "width": 2}, name="Reference")
        steeringScope.axes[4].plot.addItem(referencePath)
        referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])

        steeringScope.axes[4].attachSignal(name="Estimated", width=2)

        arrow = pg.ArrowItem(angle=180, tipAngle=60, headLen=10, tailLen=10, tailWidth=5, pen={"color": "w", "fillColor": [196, 78, 82], "width": 1}, brush=[196, 78, 82])
        arrow.setPos(initialPose[0], initialPose[1])
        steeringScope.axes[4].plot.addItem(arrow)
    # endregion

    # region : Setup control thread, then run experiment
    controlThread = Thread(target=controlLoop)
    controlThread.start()

    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
    # endregion
    
    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()

    input("Experiment complete. Press any key to exit...")
# endregion