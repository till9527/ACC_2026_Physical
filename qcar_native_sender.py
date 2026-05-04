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

enableSteeringControl = True
K_stanley = 1

nodeSequence = [
    10, 2, 4, 14, 16, 18, 11, 12, 8, 10
]

# --- Global variables ---
is_running = True
latest_frame = None
frame_lock = threading.Lock()

# Define the car's state locally since we are no longer receiving commands
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


# --- Thread Functions ---
def camera_thread_func(camera):
    global latest_frame, is_running
    print("Camera thread started...")
    
    # Hardcode the threshold value that was default in the UI
    threshold_value = 115 
    
    while is_running:
        if camera.read():
            # 1. Get raw frame
            raw_frame = camera.imageData

            # 2. CROP TO LOWER HALF (to match computer_receiver_opengl.py)
            h, w = raw_frame.shape[:2]
            lower_half_frame = raw_frame[int(h / 2):h, :]

            # 3. CLEAN THRESHOLDING
            # Convert to Grayscale
            gray = cv2.cvtColor(lower_half_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian Blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Binary Threshold (using 115)
            _, binaryImage = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

            with frame_lock:
                latest_frame = binaryImage
                cv2.imshow("Processed Frame", binaryImage)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    is_running = False

        time.sleep(1 / FRAME_RATE)
    print("Camera thread stopped.")


def control_thread_func(initialPose, waypointSequence, calibrationPose, calibrate):
    global is_running
    print("Control thread started...")

    speedController = SpeedController(kp=K_p, ki=K_i)
    if enableSteeringControl:
        steeringController = SteeringController(waypoints=waypointSequence, k=K_stanley)

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
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
                    y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
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

                # === Steering Application (GPS Only) ===
                if enableSteeringControl:
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
            print("\n[ERROR] Path generation failed! Tracing nodeSequence for errors...")
            sys.exit(1)

        initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
    else:
        initialPose = [0, 0, 0]

    calibrate = "y" in input("Do you want to recalibrate? (y/n): ")
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

        # The main thread just sleeps while the camera and control threads run in the background
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