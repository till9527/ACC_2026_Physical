import time
import signal
import numpy as np
from threading import Thread

# Import QCar and Quanser APIs
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from hal.content.qcar_functions import QCarEKF

# Setup virtual environment if not using a physical car
if not IS_PHYSICAL_QCAR:
    import qlabs_setup

global KILL_THREAD
KILL_THREAD = False

current_x = 0.0
current_y = 0.0
is_ready = False  # Flag to let us know when we have a real coordinate


def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True


signal.signal(signal.SIGINT, sig_handler)


def sensor_loop():
    global KILL_THREAD, current_x, current_y, is_ready

    qcar = QCar(readMode=1, frequency=100)
    # calibrate=False ensures it relies entirely on the existing stored workspace calibration
    gps = QCarGPS(calibrate=False)

    ekf_initialized = False
    ekf = None

    with qcar, gps:
        t0 = time.time()
        t = 0
        while not KILL_THREAD:
            tp = t
            t = time.time() - t0
            dt = t - tp

            qcar.read()

            # Read GPS and update the Kalman Filter
            if gps.readGPS():
                y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])

                # Seed the EKF with the actual physical position from the very first GPS read
                if not ekf_initialized:
                    ekf = QCarEKF(x_0=[y_gps[0], y_gps[1], y_gps[2]])
                    ekf_initialized = True
                    is_ready = True

                ekf.update([qcar.motorTach, 0], dt, y_gps, qcar.gyroscope[2])
            else:
                if ekf_initialized:
                    ekf.update([qcar.motorTach, 0], dt, None, qcar.gyroscope[2])

            if ekf_initialized:
                current_x = ekf.x_hat[0, 0]
                current_y = ekf.x_hat[1, 0]

            # Keep the car stationary
            qcar.write(0, 0)

        qcar.read_write_std(throttle=0, steering=0)


if __name__ == "__main__":
    # Note: If you are in the virtual environment (QLabs), this setup block
    # will teleport the car to [0,0,0]. If you want it to stay where you
    # previously left it in QLabs, you might need to comment this out.
    if not IS_PHYSICAL_QCAR:
        qlabs_setup.setup(initialPosition=[0, 0, 0], initialOrientation=[0, 0, 0])

    print("Initializing QCar sensors and grabbing stored calibration...")

    sensor_thread = Thread(target=sensor_loop)
    sensor_thread.start()

    # Wait until the EKF is seeded with the real GPS coordinates
    while not is_ready and not KILL_THREAD:
        time.sleep(0.1)

    print("\n--- QCar Coordinate Tracker ---")
    print("Type 'p' and press ENTER to print coordinates.")
    print("Type 'q' and press ENTER to quit.\n")

    try:
        while not KILL_THREAD:
            user_input = input().strip().lower()

            if user_input == "p":
                print(f"📍 Current Position -> X: {current_x:.4f}, Y: {current_y:.4f}")
            elif user_input == "q":
                print("Exiting...")
                KILL_THREAD = True
    except KeyboardInterrupt:
        KILL_THREAD = True

    sensor_thread.join()
    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()
