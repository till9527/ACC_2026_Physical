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
import threading
from pal.products.qcar import QCar, QCarGPS, QCarCameras, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.vision import Camera2D
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images
from custom_roadmap import CustomRoadMap
CAMERA_ID = "3"

# ================ Experiment Configuration ================
# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500
tf = 6000
startDelay = 1
controllerUpdateRate = 100

# ===== Speed Controller Parameters
# - v_ref: desired velocity in m/s
# - K_p: proportional gain for speed controller
# - K_i: integral gain for speed controller
v_ref = 0.5
K_p = 0.1
K_i = 1

# ===== Steering Controller Parameters
# - enableSteeringControl: whether or not to enable steering control
# - K_stanley: K gain for stanley controller
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
enableSteeringControl = True
K_stanley = 1

# ===== Vision Controller Parameters
# - enableVisionControl: enables the hybrid vision/GPS system
# - vision_kp: Proportional gain for vision steering
enableVisionControl = True
vision_kp = 0.002
frame_lock = threading.Lock()

nodeSequence = [
    # Lap 1: The Grand Outer Tour
    10, 2, 4, 6, 8, 23, 21, 16, 18, 11, 12, 7, 14, 20, 22, 9, 13, 19, 17, 15, 5, 3, 1, 8,
    # Lap 2: Inner City & Left-Side Crossings
    10, 1, 7, 5, 3, 1, 13, 19, 17, 20, 22,
    # Lap 3: Mid-Track Cuts & Reverse Roundabout Entry
    10, 2, 4, 14, 16, 17, 15, 6, 0, 2, 4, 6, 13, 19, 17, 16, 18, 11, 12, 8,
    # Lap 4: The Cleanup Lap
    10, 1, 8, 23, 21, 16, 18, 11, 12, 0, 2, 4, 14, 20, 22, 9, 7, 14, 20, 22, 9, 0, 2, 4, 6, 8, 10,
]


# endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# region : Initial setup
if enableSteeringControl:
    roadmap = CustomRoadMap()
    waypointSequence = roadmap.generate_path(nodeSequence)

    # --- NEW DEBUGGING BLOCK ---
    if waypointSequence is None:
        print("\n[ERROR] Path generation failed! Tracing nodeSequence for errors...")
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

        import sys

        sys.exit(1)  # Stop the program before it crashes PyQTGraph
    # ---------------------------

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

# Define the calibration pose
# Calibration pose is either [0,0,-pi/2] or [0,2,-pi/2]
# Comment out the one that is not used
# calibrationPose = [0,0,-np.pi/2]
calibrationPose = [0, 2, -np.pi / 2]

# Used to enable safe keyboard triggered shutdown
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

    # ==============  SECTION A -  Speed Control  ====================
    def update(self, v, v_ref, dt):

        e = v_ref - v
        self.ei += dt * e

        return np.clip(
            self.kp * e + self.ki * self.ei, -self.maxThrottle, self.maxThrottle
        )


class SteeringController:

    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi / 6

        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0

        self.k = k
        self.cyclic = cyclic

        self.p_ref = (0, 0)
        self.th_ref = 0

    # ==============  SECTION B -  Steering Control  ====================
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
        dist_to_next = np.linalg.norm(p - wp_2)
        if s >= v_mag or dist_to_next < 0.1:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1

        ep = wp_1 + v_uv * s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent - th)

        self.p_ref = ep
        self.th_ref = tangent

        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k * ect, speed)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle,
        )


class VisionSteeringController:

    def __init__(self, kp=0.005):
        self.kp = kp
        self.maxSteeringAngle = np.pi / 6

    def process_image(self, image):
        # 1. Crop the image to only look at the bottom half (the road ahead)
        h, w = image.shape[:2]
        roi = image[int(h / 2) : h, :]

        # 2. Convert to HSV color space for easier color isolation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 3. Define color range for the yellow lane line
        # NOTE: You WILL need to tune these values based on your room's lighting!
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        # Create a mask that only shows the yellow line
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 4. Find the center of the detected line using image moments
        M = cv2.moments(mask)
        if M["m00"] > 0:
            # Calculate the X coordinate of the center of the line
            cx = int(M["m10"] / M["m00"])
            
            # Calculate error: distance from the center of the line to the center of the camera
            error = (w / 2) - cx 
            return error, mask
        
        # If no line is found, return 0 error
        return 0, mask

    def update(self, image):
        # Get the pixel error from the center
        error, mask = self.process_image(image)

        # Simple Proportional (P) control for steering
        # If line is to the right (negative error), steer right (negative angle)
        steering_cmd = self.kp * error 

        return np.clip(steering_cmd, -self.maxSteeringAngle, self.maxSteeringAngle), mask


def controlLoop(camera):
    # region controlLoop setup
    global KILL_THREAD
    u = 0
    delta = 0
    # used to limit data sampling to 10hz
    countMax = controllerUpdateRate / 10
    count = 0
    # endregion

    # region Controller initialization
    speedController = SpeedController(kp=K_p, ki=K_i)
    if enableSteeringControl:
        steeringController = SteeringController(waypoints=waypointSequence, k=K_stanley)
        
    if enableVisionControl:
        visionController = VisionSteeringController(kp=vision_kp)
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
            
            front_image = None
            if enableVisionControl and camera is not None:
                if camera.read():
                    with frame_lock:
                        front_image = camera.imageData

            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array(
                        [gps.position[0], gps.position[1], gps.orientation[2]]
                    )
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        y_gps,
                        qcar.gyroscope[2],
                    )
                else:
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        None,
                        qcar.gyroscope[2],
                    )

                x = ekf.x_hat[0, 0]
                y = ekf.x_hat[1, 0]
                th = ekf.x_hat[2, 0]
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            v = qcar.motorTach
            # endregion

            # region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0
            else:
                # region : Speed controller update
                u = speedController.update(v, v_ref, dt)
                # endregion

                # region : Steering controller update
                if enableSteeringControl:
                    # Always calculate GPS delta as a baseline
                    gps_delta = steeringController.update(p, th, v)
                    
                    if enableVisionControl and front_image is not None:
                        vision_delta, debug_mask = visionController.update(front_image)
                        
                        # --- HYBRID BLENDING LOGIC ---
                        # Count how many yellow pixels the camera sees. 
                        # If it sees a solid line, use vision. If the line breaks 
                        # (like at an intersection), fall back to the GPS controller!
                        if cv2.countNonZero(debug_mask) > 50:
                            delta = vision_delta
                        else:
                            delta = gps_delta
                            
                        # UNCOMMENT THESE TWO LINES TO DEBUG LIGHTING/COLORS
                        cv2.imshow("Lane Mask", debug_mask)
                        cv2.waitKey(1)
                    else:
                        delta = gps_delta
                else:
                    delta = 0
                # endregion

            qcar.write(u, delta)
            # endregion

            # region : Update Scopes
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay

                # Speed control scope
                speedScope.axes[0].sample(t_plot, [v, v_ref])
                speedScope.axes[1].sample(t_plot, [v_ref - v])
                speedScope.axes[2].sample(t_plot, [u])

                # Steering control scope
                if enableSteeringControl:
                    steeringScope.axes[4].sample(t_plot, [[p[0], p[1]]])

                    p[0] = ekf.x_hat[0, 0]
                    p[1] = ekf.x_hat[1, 0]

                    x_ref = steeringController.p_ref[0]
                    y_ref = steeringController.p_ref[1]
                    th_ref = steeringController.th_ref

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

    # Initialize the camera on the MAIN thread to avoid Quanser API affinity errors
    if enableVisionControl:
        camera = Camera2D(
            cameraId=CAMERA_ID,
            frameWidth=640,
            frameHeight=480,
            frameRate=30,
        )
    else:
        camera = None

    # region : Setup scopes
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30
    # Scope for monitoring speed controller
    speedScope = MultiScope(rows=3, cols=1, title="Vehicle Speed Control", fps=fps)
    speedScope.addAxis(
        row=0, col=0, timeWindow=tf, yLabel="Vehicle Speed [m/s]", yLim=(0, 1)
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

    # Scope for monitoring steering controller
    if enableSteeringControl:
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
            row=3, col=0, timeWindow=tf, yLabel="Steering Angle [rad]", yLim=(-0.6, 0.6)
        )
        steeringScope.axes[3].attachSignal()
        steeringScope.axes[3].xLabel = "Time [s]"

        steeringScope.addXYAxis(
            row=0,
            col=1,
            rowSpan=4,
            xLabel="x Position [m]",
            yLabel="y Position [m]",
            xLim=(-2.5, 2.5),
            yLim=(-1, 5),
        )

        im = cv2.imread(images.SDCS_CITYSCAPE, cv2.IMREAD_GRAYSCALE)

        steeringScope.axes[4].attachImage(
            scale=(-0.002035, 0.002035),
            offset=(1125, 2365),
            rotation=180,
            levels=(0, 255),
        )
        steeringScope.axes[4].images[0].setImage(image=im)

        referencePath = pg.PlotDataItem(
            pen={"color": (85, 168, 104), "width": 2}, name="Reference"
        )
        steeringScope.axes[4].plot.addItem(referencePath)
        referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])

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
        arrow.setPos(initialPose[0], initialPose[1])
        steeringScope.axes[4].plot.addItem(arrow)
    # endregion

    # region : Setup control thread, then run experiment
    # Pass the main-thread-initialized camera into the control loop!
    controlThread = Thread(target=controlLoop, args=(camera,))
    controlThread.start()

    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
        # Safely release the camera to prevent ghost locks on next run
        if enableVisionControl and camera is not None:
            camera.terminate()
            
    # endregion
    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()

    input("Experiment complete. Press any key to exit...")
# endregion