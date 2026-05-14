# qcar_standalone_control.py - Runs ON THE QCAR

import threading
import time
import signal
import numpy as np
import cv2
import sys

from pal.products.qcar import (
    QCar,
    QCarGPS,
    IS_PHYSICAL_QCAR,
)
from pal.utilities.vision import Camera2D
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.utilities.image_processing import ImageProcessing
from custom_roadmap import CustomRoadMap

# --- Camera Settings ---
CAMERA_ID = "2"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FRAME_RATE = 30

# --- Controller Settings ---
tf = 6000
startDelay = 1
controllerUpdateRate = 100
v_ref = 0.5
K_p = 0.1
K_i = 1

# Toggle this to False to go back to GPS waypoint navigation
USE_VISION_STEERING = True

enableSteeringControl = True
K_stanley = 1

nodeSequence = [10, 2, 4, 14, 16, 18, 11, 12, 8, 10]

# --- Global variables ---
is_running = True
latest_frame = None
frame_lock = threading.Lock()

# Define the car's state locally
car_state = "FORCE_GO"


# --- Shutdown Signal Handler ---
def sig_handler(*args):
    global is_running
    is_running = False
    print("\nShutdown signal received.")


signal.signal(signal.SIGINT, sig_handler)


# --- Controller Classes ---
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
    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic
        self.p_ref = (0, 0)
        self.th_ref = 0

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
    """
    PD Controller that uses the centroid of the binary mask to keep the car centered.
    """

    def __init__(self, kp=0.003, kd=0.001):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0
        self.maxSteeringAngle = np.pi / 6

    def update(self, binary_mask, dt):
        if binary_mask is None:
            return 0.0

        h, w = binary_mask.shape

        # Look at a horizontal strip in the lower middle of the mask
        # This prevents the car from being distracted by distant lines or the extreme edges
        strip = binary_mask[int(h * 0.5) : int(h * 0.9), :]

        # Calculate image moments to find the centroid of the white pixels
        M = cv2.moments(strip)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])

            target_center = w / 2

            # If centroid is right of center (cx > target_center), error is negative -> steer right
            error = target_center - cx

            derivative = (error - self.prev_error) / dt if dt > 0 else 0
            self.prev_error = error

            steering = (self.kp * error) + (self.kd * derivative)
            return np.clip(steering, -self.maxSteeringAngle, self.maxSteeringAngle)

        # If no lines are seen, keep the wheels straight
        return 0.0


# --- Thread Functions ---
def camera_thread_func(camera):
    global latest_frame, is_running
    print("Camera thread started...")

    threshold_value = 115

    while is_running:
        if camera.read():
            raw_frame = camera.imageData

            # Crop to lower half to match original UI logic
            h, w = raw_frame.shape[:2]
            lower_half_frame = raw_frame[int(h / 2) : h, :]

            # Grayscale, blur, and threshold
            gray = cv2.cvtColor(lower_half_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binaryImage = cv2.threshold(
                blurred, threshold_value, 255, cv2.THRESH_BINARY
            )

            with frame_lock:
                latest_frame = binaryImage

        time.sleep(1 / FRAME_RATE)
    print("Camera thread stopped.")


def control_thread_func(initialPose, waypointSequence, calibrationPose, calibrate):
    global is_running
    print("Control thread started...")

    speedController = SpeedController(kp=K_p, ki=K_i)

    # Initialize both controllers
    if enableSteeringControl:
        steeringController = SteeringController(waypoints=waypointSequence, k=K_stanley)
    visionSteeringController = VisionSteeringController(kp=0.003, kd=0.001)

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)

    # We still initialize EKF/GPS so the car doesn't throw errors if it's plugged in,
    # but we will bypass its steering output if USE_VISION_STEERING is True.
    if enableSteeringControl:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)
    else:
        gps = memoryview(b"")

    with qcar, gps:
        t0 = time.time()
        t = 0
        delta = 0
        while (t < tf + startDelay) and is_running:
            tp = t
            t = time.time() - t0
            dt = t - tp

            qcar.read()
            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array(
                        [gps.position[0], gps.position[1], gps.orientation[2]]
                    )
                    ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
                else:
                    ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])
                x, y, th = ekf.x_hat[0, 0], ekf.x_hat[1, 0], ekf.x_hat[2, 0]
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            v = qcar.motorTach

            if t < startDelay:
                u, delta = 0, 0
            else:
                state = car_state

                # === Speed Application ===
                if state == "FORCE_GO":
                    target_speed = v_ref
                elif state == "FORCE_STOP" or state == "STOP":
                    target_speed = 0.0
                else:
                    target_speed = v_ref

                u = speedController.update(v, target_speed, dt)

                # === Steering Application ===
                if USE_VISION_STEERING:
                    # Grab the latest binary mask safely
                    local_mask = None
                    with frame_lock:
                        if latest_frame is not None:
                            local_mask = latest_frame.copy()

                    # Calculate steering angle based on lane lines
                    delta = visionSteeringController.update(local_mask, dt)

                elif enableSteeringControl:
                    # Fallback to standard GPS tracking
                    delta = steeringController.update(p, th, v)
                else:
                    delta = 0

            qcar.write(u, delta)

        qcar.write(0, 0)
    is_running = False
    print("Control thread stopped.")


# --- Main Program ---
if not IS_PHYSICAL_QCAR:
    print("This script is designed to run on the physical QCar.")
else:
    if enableSteeringControl:
        roadmap = CustomRoadMap()
        waypointSequence = roadmap.generate_path(nodeSequence)

        if waypointSequence is None:
            print(
                "\n[ERROR] Path generation failed! Tracing nodeSequence for errors..."
            )
            sys.exit(1)

        initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
    else:
        initialPose = [0, 0, 0]

    # Only ask to recalibrate GPS if we are actually using it
    calibrate = False
    if not USE_VISION_STEERING:
        calibrate = "y" in input("Do you want to recalibrate GPS? (y/n): ")

    calibrationPose = [0, 2, -np.pi / 2]

    controlThread = None
    cameraThread = None
    camera = None

    try:
        camera = Camera2D(
            cameraId=CAMERA_ID,
            frameWidth=IMAGE_WIDTH,
            frameHeight=IMAGE_HEIGHT,
            frameRate=FRAME_RATE,
        )

        cameraThread = threading.Thread(target=camera_thread_func, args=(camera,))
        controlThread = threading.Thread(
            target=control_thread_func,
            args=(initialPose, waypointSequence, calibrationPose, calibrate),
        )

        cameraThread.start()
        controlThread.start()

        # The main thread sleeps while camera and control run
        while is_running:
            time.sleep(0.5)

    except Exception as e:
        print(f"An error occurred in the main thread: {e}")
    finally:
        print("Cleaning up resources...")
        is_running = False

        if controlThread and controlThread.is_alive():
            controlThread.join()
        if cameraThread and cameraThread.is_alive():
            cameraThread.join()

        if camera:
            camera.terminate()

        print("Shutdown complete.")
