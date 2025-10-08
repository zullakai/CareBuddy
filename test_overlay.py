import cv2
import time
import psutil
import torch
from collections import deque

# Check for GPU
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# Initialize camera
cap = cv2.VideoCapture(0)

# === Variables ===
fps_values = deque(maxlen=30)  # store recent FPS for smoothing
last_fps_update = time.time()
display_fps = 0

last_sys_update = time.time()
cpu_usage = 0
memory_usage = 0

# === Main Loop ===
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # --- FPS Calculation ---
    fps = 1 / (time.time() - start_time)
    fps_values.append(fps)

    # Update displayed FPS every 1 sec
    if time.time() - last_fps_update > 1:
        display_fps = sum(fps_values) / len(fps_values)
        last_fps_update = time.time()

    # --- CPU & RAM every 2 sec ---
    if time.time() - last_sys_update > 2:
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent
        last_sys_update = time.time()

    # --- Overlay Info ---
    cv2.putText(frame, f'FPS: {display_fps:.1f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f'CPU: {cpu_usage:.1f}%', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f'MEM: {memory_usage:.1f}%', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f'DEVICE: {device_name}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Show window
    cv2.imshow("CareBuddy Performance Monitor", frame)

    # --- Console logging (optional for testing) ---
    # print(f"FPS: {display_fps:.1f}, CPU: {cpu_usage:.1f}%, MEM: {memory_usage:.1f}%")

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
