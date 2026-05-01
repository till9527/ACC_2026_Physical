# computer_receiver_opengl.py - Runs ON YOUR COMPUTER

import socket
import struct
import numpy as np
import cv2
import threading

# --- Settings ---
HOST = "0.0.0.0"
PORT = 8080

# --- Globals ---
latest_frames = {}
latest_masks = {}  
frames_lock = threading.Lock()

# Dictionary to control client overrides from the main thread
client_controls = {}
controls_lock = threading.Lock()

# NEW: Global threshold value for live tuning
live_threshold = 115

def receive_all(sock, n):
    data = bytearray()
    while len(data) < n:
        try:
            sock.settimeout(5.0)
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        except (socket.timeout, socket.error):
            return None
    return data

def handle_client(conn, addr):
    """
    Receives frames, crops to the lower half, generates a clean binary mask, 
    and sends basic state commands back to the QCar.
    """
    global live_threshold
    print(f"Thread started for client: {addr}")

    with controls_lock:
        client_controls[addr] = {"force_go": False, "force_stop": False}

    payload_size = struct.calcsize(">L")
    car_state = "GO"

    try:
        while True:
            # --- 1. Receive Frame ---
            packed_msg_size = receive_all(conn, payload_size)
            if not packed_msg_size:
                break
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            frame_data = receive_all(conn, msg_size)
            if not frame_data:
                break

            np_arr = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # --- 2. CROP TO LOWER HALF ---
            h, w = frame.shape[:2]
            lower_half_frame = frame[int(h / 2):h, :]

            # --- 3. CLEAN THRESHOLDING (Live Tunable) ---
            gray = cv2.cvtColor(lower_half_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use the global live_threshold variable instead of a hardcoded number!
            _, cropped_mask = cv2.threshold(blurred, live_threshold, 255, cv2.THRESH_BINARY)

            # --- 4. Check Override States ---
            force_go_active = False
            force_stop_active = False
            with controls_lock:
                if addr in client_controls:
                    force_go_active = client_controls[addr]["force_go"]
                    force_stop_active = client_controls[addr]["force_stop"]

            prev_car_state = car_state

            # Draw text on the cropped frame
            if force_stop_active:
                car_state = "FORCE_STOP"
                cv2.putText(lower_half_frame, "MANUAL: STOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif force_go_active:
                car_state = "FORCE_GO"
                cv2.putText(lower_half_frame, "MANUAL: GO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                car_state = "GO"
                # Added threshold value to the debug text so you can see it live!
                cv2.putText(lower_half_frame, f"AUTO (Thresh: {live_threshold})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if car_state != prev_car_state:
                print(f"--- {addr}: State changed to {car_state} ---")

            # --- 5. Send the basic telemetry string ---
            cmd_msg = f"{car_state}\n"
            conn.sendall(cmd_msg.encode("utf-8"))

            # Update shared dictionary for the main thread to display
            with frames_lock:
                latest_frames[addr] = lower_half_frame
                latest_masks[addr] = cropped_mask

    except Exception as e:
        print(f"Error for {addr}: {e}")
    finally:
        print(f"Closing {addr}.")
        with frames_lock:
            if addr in latest_frames:
                del latest_frames[addr]
            if addr in latest_masks:
                del latest_masks[addr]
        with controls_lock:
            if addr in client_controls:
                del client_controls[addr]
        conn.close()


def main():
    global live_threshold
    
    def accept_connections(server_sock):
        while True:
            try:
                conn, addr = server_sock.accept()
                thread = threading.Thread(target=handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()
            except socket.error:
                break

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    
    print(f"Server listening on {PORT}.")
    print("--- CONTROLS ---")
    print("Press 'g' to FORCE GO")
    print("Press 's' to FORCE STOP")
    print("Press 'a' to switch back to AUTO (GPS)")
    print("Press ']' to INCREASE threshold (less white)")
    print("Press '[' to DECREASE threshold (more white)")
    print("Press 'q' to QUIT")
    print("----------------")

    accept_thread = threading.Thread(target=accept_connections, args=(server_socket,))
    accept_thread.daemon = True
    accept_thread.start()

    active_windows = set()
    running = True
    try:
        while running:
            with frames_lock:
                frames_to_show = latest_frames.copy()
                masks_to_show = latest_masks.copy()

            current_windows = set()
            
            # Show Standard Camera Feed
            for addr, frame in frames_to_show.items():
                window_name = f"QCar Feed {addr[0]}"
                cv2.imshow(window_name, frame)
                current_windows.add(window_name)
                active_windows.add(window_name)
            
            # Show Pure Black and White Mask
            for addr, mask in masks_to_show.items():
                window_name_mask = f"Lane Detection View {addr[0]}"
                cv2.imshow(window_name_mask, mask)
                current_windows.add(window_name_mask)
                active_windows.add(window_name_mask)

            windows_to_close = active_windows - current_windows
            for window_name in windows_to_close:
                cv2.destroyWindow(window_name)
            active_windows = current_windows

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                running = False
            elif key == ord("g"):
                print(">>> UI COMMAND: Enabling Force GO")
                with controls_lock:
                    for addr in client_controls:
                        client_controls[addr]["force_go"] = True
                        client_controls[addr]["force_stop"] = False
            elif key == ord("s"):
                print(">>> UI COMMAND: Enabling Force STOP")
                with controls_lock:
                    for addr in client_controls:
                        client_controls[addr]["force_stop"] = True
                        client_controls[addr]["force_go"] = False
            elif key == ord("a"):
                print(">>> UI COMMAND: Re-enabling AUTO Mode")
                with controls_lock:
                    for addr in client_controls:
                        client_controls[addr]["force_go"] = False
                        client_controls[addr]["force_stop"] = False
            elif key == ord("]"):
                live_threshold = min(255, live_threshold + 5)
                print(f">>> TUNING: Threshold increased to {live_threshold}")
            elif key == ord("["):
                live_threshold = max(0, live_threshold - 5)
                print(f">>> TUNING: Threshold decreased to {live_threshold}")

            if not accept_thread.is_alive() and not active_windows:
                if not accept_thread.is_alive():
                    running = False

    finally:
        print("\nShutting down.")
        server_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()