#!/usr/bin/env python3
import time
import numpy as np
import cv2
import signal
from threading import Thread
import pyqtgraph as pg

# --- Quanser Hardware & Math Imports ---
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from hal.content.qcar_functions import QCarEKF
from pal.utilities.math import wrap_to_pi
from pal.utilities.scope import MultiScope
import pal.resources.images as images

# --- Quanser Standard Map Import ---
from hal.products.mats import SDCSRoadMap

# =========================================================
# CONFIGURATION
# =========================================================
BASE_SPEED = 0.5  # m/s (Start slow on the real car!)
MAX_STEERING = np.pi / 6  # 30 degrees

# Max experiment duration and startup delay
tf = 6000
startDelay = 1
controllerUpdateRate = 100

# Default sequence for the standard Quanser SDCSRoadMap
NODE_SEQUENCE = [
    10,
    2,
    4,
    14,
    16,
    18,
    11,
    12,
    7,
    5,
    3,
    1,
    8,
    23,
    21,
    16,
    17,
    20,
    22,
    9,
    0,
    2,
    4,
    6,
    13,
    19,
    17,
    15,
    6,
    0,
    2,
    4,
    6,
    8,
    10,
]

# Controller Tuning
K_STANLEY = 1.0
K_P_SPEED = 0.1
K_I_SPEED = 1.0

# Used to safely shutdown the background thread
global KILL_THREAD
KILL_THREAD = False


def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True


signal.signal(signal.SIGINT, sig_handler)


# =========================================================
# CONTROLLERS
# =========================================================
class SpeedController:
    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        return np.clip(
            self.kp * e + self.ki * self.ei, -self.maxThrottle, self.maxThrottle
        )


class SteeringController:
    def __init__(self, waypoints, k=1, cyclic=False):
        self.maxSteeringAngle = MAX_STEERING
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic

    def update(self, p, th, speed):
        wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
        wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]

        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except ZeroDivisionError:
            return 0

        tangent = np.arctan2(v_uv[1], v_uv[0])
        s = np.dot(p - wp_1, v_uv)

        # Move to next waypoint if we pass the current segment
        if s >= v_mag:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1

        ep = wp_1 + v_uv * s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent - th)

        # Stanley Control Law
        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k * ect, speed + 0.001)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle,
        )


# =========================================================
# BACKGROUND THREAD (Hardware Control Loop)
# =========================================================
def controlLoop():
    global KILL_THREAD

    # Setup Controllers and EKF
    speedController = SpeedController(kp=K_P_SPEED, ki=K_I_SPEED)
    steeringController = SteeringController(
        waypoints=waypointSequence, k=K_STANLEY, cyclic=False
    )

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    ekf = QCarEKF(x_0=actual_pose)

    print("\n--- STARTING HARDWARE RUN ---")
    print("Press Ctrl+C to stop.")

    # Scope sampling limits (10Hz sampling instead of 100Hz to save CPU)
    countMax = controllerUpdateRate / 10
    count = 0

    with qcar, gps:
        t0 = time.time()
        t = 0
        throttle = 0.0
        steering = 0.0

        try:
            while (t < tf + startDelay) and (not KILL_THREAD):
                # Timing
                tp = t
                t = time.time() - t0
                dt = t - tp

                # Read Sensors
                qcar.read()

                # Update EKF
                if gps.readGPS():
                    y_gps = np.array(
                        [gps.position[0], gps.position[1], gps.orientation[2]]
                    )
                    ekf.update([qcar.motorTach, steering], dt, y_gps, qcar.gyroscope[2])
                else:
                    ekf.update([qcar.motorTach, steering], dt, None, qcar.gyroscope[2])

                # Current State
                x = ekf.x_hat[0, 0]
                y = ekf.x_hat[1, 0]
                th = ekf.x_hat[2, 0]
                current_speed = qcar.motorTach
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2

                # Controller Updates
                if t < startDelay:
                    throttle = 0.0
                    steering = 0.0
                else:
                    throttle = speedController.update(current_speed, BASE_SPEED, dt)
                    steering = steeringController.update(p, th, current_speed)

                qcar.write(throttle, steering)

                # Update Scopes (Data Collection)
                count += 1
                if count >= countMax and t > startDelay:
                    t_plot = t - startDelay

                    # Speed Scope
                    speedScope.axes[0].sample(t_plot, [current_speed, BASE_SPEED])
                    speedScope.axes[1].sample(t_plot, [BASE_SPEED - current_speed])
                    speedScope.axes[2].sample(t_plot, [throttle])

                    # Steering Scope
                    steeringScope.axes[0].sample(t_plot, [p[0], gps.position[0]])
                    steeringScope.axes[1].sample(t_plot, [p[1], gps.position[1]])
                    steeringScope.axes[2].sample(t_plot, [th, gps.orientation[2]])
                    steeringScope.axes[3].sample(t_plot, [steering])

                    # Map Plot (Draws the green path and the moving arrow)
                    steeringScope.axes[4].sample(t_plot, [[p[0], p[1]]])
                    arrow.setPos(p[0], p[1])
                    arrow.setStyle(angle=180 - th * 180 / np.pi)

                    count = 0

        finally:
            print("[INFO] Stopping motors.")
            qcar.read_write_std(throttle=0, steering=0)


# =========================================================
# MAIN THREAD (Map Generation & UI)
# =========================================================
if __name__ == "__main__":
    if not IS_PHYSICAL_QCAR:
        print("[ERROR] This script is meant for the physical QCar!")
        exit()

    # 1. Setup Roadmap
    print("Generating Path from Quanser SDCSRoadMap...")
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(NODE_SEQUENCE)
    theoretical_start = roadmap.get_node_pose(NODE_SEQUENCE[0]).squeeze()

    # 2. Initialize GPS
    print("Connecting to GPS to measure actual offset...")
    gps = QCarGPS(initialPose=theoretical_start, calibrate=True)
    actual_pose = None

    with gps:
        for _ in range(50):
            if gps.readGPS():
                actual_pose = np.array(
                    [gps.position[0], gps.position[1], gps.orientation[2]]
                )
            time.sleep(0.01)

    if actual_pose is None:
        print(
            "[WARNING] Could not get GPS fix. Falling back to theoretical coordinates."
        )
        actual_pose = theoretical_start
    else:
        print(f"-> Theoretical Node 0: {theoretical_start}")
        print(f"-> Actual GPS Pose:    {actual_pose}")

    # 3. Dynamic Coordinate Calibration
    print("Calibrating waypoint map to actual position...")
    dtheta = actual_pose[2] - theoretical_start[2]

    shifted_x = waypointSequence[0, :] - theoretical_start[0]
    shifted_y = waypointSequence[1, :] - theoretical_start[1]
    cos_t, sin_t = np.cos(dtheta), np.sin(dtheta)

    rotated_x = shifted_x * cos_t - shifted_y * sin_t
    rotated_y = shifted_x * sin_t + shifted_y * cos_t

    waypointSequence[0, :] = rotated_x + actual_pose[0]
    waypointSequence[1, :] = rotated_y + actual_pose[1]

    # 4. Configure pyqtgraph Scopes
    fps = 10
    speedScope = MultiScope(rows=3, cols=1, title="Vehicle Speed Control", fps=fps)
    speedScope.addAxis(
        row=0, col=0, timeWindow=tf, yLabel="Vehicle Speed [m/s]", yLim=(0, 1.5)
    )
    speedScope.axes[0].attachSignal(name="v_meas", width=2)
    speedScope.axes[0].attachSignal(name="v_ref")
    speedScope.addAxis(
        row=1, col=0, timeWindow=tf, yLabel="Speed Error [m/s]", yLim=(-0.5, 0.5)
    )
    speedScope.axes[1].attachSignal()
    speedScope.addAxis(
        row=2,
        col=0,
        timeWindow=tf,
        xLabel="Time [s]",
        yLabel="Throttle Command [%]",
        yLim=(-0.3, 0.3),
    )
    speedScope.axes[2].attachSignal()

    steeringScope = MultiScope(
        rows=4, cols=2, title="Vehicle Steering Control", fps=fps
    )
    steeringScope.addAxis(
        row=0, col=0, timeWindow=tf, yLabel="x Position [m]", yLim=(-2.5, 2.5)
    )
    steeringScope.axes[0].attachSignal(name="x_meas")
    steeringScope.axes[0].attachSignal(name="x_ref")
    steeringScope.addAxis(
        row=1, col=0, timeWindow=tf, yLabel="y Position [m]", yLim=(-1, 5)
    )
    steeringScope.axes[1].attachSignal(name="y_meas")
    steeringScope.axes[1].attachSignal(name="y_ref")
    steeringScope.addAxis(
        row=2, col=0, timeWindow=tf, yLabel="Heading Angle [rad]", yLim=(-3.5, 3.5)
    )
    steeringScope.axes[2].attachSignal(name="th_meas")
    steeringScope.axes[2].attachSignal(name="th_ref")
    steeringScope.addAxis(
        row=3,
        col=0,
        timeWindow=tf,
        xLabel="Time [s]",
        yLabel="Steering Angle [rad]",
        yLim=(-0.6, 0.6),
    )
    steeringScope.axes[3].attachSignal()

    # Map Plotting Area
    steeringScope.addXYAxis(
        row=0,
        col=1,
        rowSpan=4,
        xLabel="x Position [m]",
        yLabel="y Position [m]",
        xLim=(-2.5, 2.5),
        yLim=(-1, 5),
    )

    # Load SDCS Cityscape Background Image
    im = cv2.imread(images.SDCS_CITYSCAPE, cv2.IMREAD_GRAYSCALE)
    steeringScope.axes[4].attachImage(
        scale=(-0.002035, 0.002035), offset=(1125, 2365), rotation=180, levels=(0, 255)
    )
    steeringScope.axes[4].images[0].setImage(image=im)

    # Draw the green reference line
    referencePath = pg.PlotDataItem(
        pen={"color": (85, 168, 104), "width": 2}, name="Reference"
    )
    steeringScope.axes[4].plot.addItem(referencePath)
    referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])

    # Draw the moving car as an Arrow
    steeringScope.axes[4].attachSignal(name="Estimated", width=2)
    arrow = pg.ArrowItem(
        angle=180,
        tipAngle=60,
        headLen=10,
        tailLen=10,
        tailWidth=5,
        pen={"color": "w", "fillColor": [196, 78, 82], "width": 1},
        brush=[196, 78, 82],
    )
    arrow.setPos(actual_pose[0], actual_pose[1])
    steeringScope.axes[4].plot.addItem(arrow)

    # 5. Start control thread, then run the UI
    controlThread = Thread(target=controlLoop)
    controlThread.start()

    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            MultiScope.refreshAll()  # Updates pyqtgraph (Must be in main thread!)
            time.sleep(0.01)
    finally:
        KILL_THREAD = True

    print("\nExperiment complete.")
