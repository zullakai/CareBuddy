#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CareBuddy Mouse + Scroll + Pinch Click (with auto-scroll zones)
"""

import cv2 as cv
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import signal

# ---------- CONFIGURABLE PARAMETERS ----------
CAM_IDX = 0
SENSITIVITY = 1.0
ALPHA = 0.25
FRAME_SLEEP = 0.005

# Angles and thresholds
ANGLE_IDX_THRESH = 22.0
ANGLE_MID_THRESH = 16.0
MIN_FINGER_LEN_PX = 35
TIP_SEP_RATIO = 0.75
BOTH_DEBOUNCE_FRAMES = 3
EXIT_DEBOUNCE_FRAMES = 3
MOVE_DEBOUNCE_FRAMES = 1

# Scroll config
SCROLL_DEADZONE = 0.003
SCROLL_FACTOR = 1400
SCROLL_MAX_PER_STEP = 120
SCROLL_DELAY = 0.25

# Auto-scroll zones
TOP_BORDER_Y = 0.15
BOTTOM_BORDER_Y = 0.85
AUTO_SCROLL_AMOUNT = 250

# Acceleration tuning
ACCEL_THRESHOLDS = [8, 25, 60]
ACCEL_VALUES = [1.0, 1.4, 1.8, 2.3]

# Pinch threshold
PINCH_THRESHOLD = 0.05  # smaller = more sensitive


# ---------- SETUP ----------
mp_hands = mp.solutions.hands
running = True

def sigint_handler(sig, frame):
    global running
    print("\n[EXIT] Ctrl+C detected.")
    running = False
signal.signal(signal.SIGINT, sigint_handler)


# ---------- HELPER FUNCTIONS ----------
def angle_between_segments(pA, pB, pC, w, h):
    v1 = np.array([(pB.x - pA.x) * w, (pB.y - pA.y) * h])
    v2 = np.array([(pC.x - pB.x) * w, (pC.y - pB.y) * h])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return abs(np.degrees(np.arccos(cosang)))

def finger_length_px(p_start, p_end, w, h):
    return np.linalg.norm(np.array([(p_end.x - p_start.x) * w, (p_end.y - p_start.y) * h]))

def get_acceleration(speed):
    if speed < ACCEL_THRESHOLDS[0]: return ACCEL_VALUES[0]
    elif speed < ACCEL_THRESHOLDS[1]: return ACCEL_VALUES[1]
    elif speed < ACCEL_THRESHOLDS[2]: return ACCEL_VALUES[2]
    else: return ACCEL_VALUES[3]


# ---------- MAIN ----------
def main():
    print("Starting CareBuddy with Pinch Click...")
    screen_w, screen_h = pyautogui.size()

    cap = cv.VideoCapture(CAM_IDX)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    prev_x, prev_y = pyautogui.position()
    prev_x, prev_y = float(prev_x), float(prev_y)

    state = "IDLE"
    prev_scroll_avg_y = None
    index_count = both_count = none_count = 0
    last_auto_scroll_time = 0
    click_active = False  # track click state

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while running:
            success, frame = cap.read()
            if not success:
                time.sleep(0.01)
                continue

            frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                # Hand not visible → go idle
                none_count += 1
                if state != "IDLE" and none_count >= EXIT_DEBOUNCE_FRAMES:
                    state = "IDLE"
                    if click_active:
                        pyautogui.mouseUp()
                        click_active = False
                    print("[State] -> IDLE")
                time.sleep(FRAME_SLEEP)
                continue

            lm = results.multi_hand_landmarks[0].landmark

            # ----- Finger state detection -----
            ang_idx = angle_between_segments(lm[6], lm[7], lm[8], w, h)
            len_idx = finger_length_px(lm[6], lm[8], w, h)
            index_straight = ang_idx and ang_idx < ANGLE_IDX_THRESH and len_idx > MIN_FINGER_LEN_PX

            ang_mid = angle_between_segments(lm[10], lm[11], lm[12], w, h)
            len_mid = finger_length_px(lm[10], lm[12], w, h)
            middle_straight = ang_mid and ang_mid < ANGLE_MID_THRESH and len_mid > MIN_FINGER_LEN_PX

            tip_sep_px = abs((lm[8].x - lm[12].x) * w)
            avg_len = (len_idx + len_mid) / 2 if (len_idx + len_mid) > 0 else 1.0
            tip_sep_ratio = tip_sep_px / (avg_len + 1e-6)

            # Pinch detection
            thumb_tip = lm[4]
            index_tip = lm[8]
            pinch_dist = math.hypot((thumb_tip.x - index_tip.x), (thumb_tip.y - index_tip.y))

            # Debounce logic
            if index_straight and middle_straight and tip_sep_ratio <= TIP_SEP_RATIO:
                both_count += 1; index_count = 0; none_count = 0
            elif index_straight and not middle_straight:
                index_count += 1; both_count = 0; none_count = 0
            else:
                none_count += 1; both_count = 0; index_count = 0

            # State transitions
            if state != 'SCROLLING' and both_count >= BOTH_DEBOUNCE_FRAMES:
                state = 'SCROLLING'; prev_scroll_avg_y = (lm[8].y + lm[12].y) / 2; print("[State] -> SCROLLING")

            if state == 'SCROLLING' and none_count >= EXIT_DEBOUNCE_FRAMES:
                state = 'IDLE'; prev_scroll_avg_y = None; print("[State] -> IDLE")

            if state != 'SCROLLING' and index_count >= MOVE_DEBOUNCE_FRAMES:
                state = 'MOVING'

            if state == 'MOVING' and none_count >= EXIT_DEBOUNCE_FRAMES:
                state = 'IDLE'

            # ----- SCROLLING -----
            if state == 'SCROLLING' and prev_scroll_avg_y is not None:
                avg_y = (lm[8].y + lm[12].y) / 2
                dy = prev_scroll_avg_y - avg_y
                if abs(dy) > SCROLL_DEADZONE:
                    scroll_amount = int(np.clip(dy * SCROLL_FACTOR, -SCROLL_MAX_PER_STEP, SCROLL_MAX_PER_STEP))
                    pyautogui.scroll(scroll_amount)
                    prev_scroll_avg_y = prev_scroll_avg_y * 0.6 + avg_y * 0.4

            # ----- MOVING -----
            elif state == 'MOVING' and index_straight:
                idx_tip = lm[8]
                mapped_x = idx_tip.x * screen_w * SENSITIVITY
                mapped_y = idx_tip.y * screen_h * SENSITIVITY
                mapped_x = np.clip(mapped_x, 0, screen_w - 1)
                mapped_y = np.clip(mapped_y, 0, screen_h - 1)

                dx, dy = mapped_x - prev_x, mapped_y - prev_y
                speed = math.hypot(dx, dy)
                accel = get_acceleration(speed)
                target_x = prev_x + dx * accel
                target_y = prev_y + dy * accel

                smoothed_x = ALPHA * target_x + (1 - ALPHA) * prev_x
                smoothed_y = ALPHA * target_y + (1 - ALPHA) * prev_y

                pyautogui.moveTo(smoothed_x, smoothed_y, duration=0)
                prev_x, prev_y = smoothed_x, smoothed_y

                # Auto-scroll zone
                now = time.time()
                if now - last_auto_scroll_time > SCROLL_DELAY:
                    if idx_tip.y < TOP_BORDER_Y:
                        pyautogui.scroll(AUTO_SCROLL_AMOUNT)
                        last_auto_scroll_time = now
                    elif idx_tip.y > BOTTOM_BORDER_Y:
                        pyautogui.scroll(-AUTO_SCROLL_AMOUNT)
                        last_auto_scroll_time = now

            # ----- PINCH CLICK -----
            if state != "IDLE":
                if pinch_dist < PINCH_THRESHOLD and not click_active:
                    pyautogui.mouseDown()
                    click_active = True
                    print("[CLICK] Down")
                elif pinch_dist >= PINCH_THRESHOLD and click_active:
                    pyautogui.mouseUp()
                    click_active = False
                    print("[CLICK] Up")
            else:
                # In IDLE mode, ensure no click is active
                if click_active:
                    pyautogui.mouseUp()
                    click_active = False

            time.sleep(FRAME_SLEEP)

    cap.release()
    print("Stopped successfully.")


if __name__ == "__main__":
    main()
