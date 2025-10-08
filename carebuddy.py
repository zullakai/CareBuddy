import os
import cv2 as cv
import copy
import csv
import numpy as np
import mediapipe as mp
import sounddevice as sd
import torch
import queue
import threading
import time
from collections import deque, Counter
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier
import google.generativeai as genai  # ✅ Gemini import

# ========================= GEMINI SETUP =========================
GEMINI_API_KEY = "AIzaSyAccIZJvoYZ7Ctrkjo7psGhLv-g1shJaus"  # 🔑 Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

def correct_with_gemini(sentence):
    try:
        prompt = (
            f"Fix the grammar and capitalization of this sentence. "
            f"Return only the corrected sentence with no explanations: '{sentence}'"
        )
        response = gemini_model.generate_content(prompt)
        corrected = response.text.strip()
        print(f"✅ Corrected Sentence: {corrected}")
        return corrected
    except Exception as e:
        print("❌ Gemini correction failed:", e)
        return sentence

# ===============================================================

# ✅ Fix Hugging Face Cache Issue
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ✅ Load Distil-Whisper Model
model_name = "distil-whisper/distil-large-v2"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)

# ✅ Load Hand Recognition Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# ✅ Load Classifiers
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# ✅ Load Labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f)]

# ✅ Open Camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

fps_calc = CvFpsCalc(buffer_len=10)
sign_history = deque(maxlen=10)
gesture_history = deque(maxlen=10)

# ✅ Audio settings
sample_rate = 16000
recording_duration = 3
energy_threshold = 10.0

# ✅ Queues
audio_queue = queue.Queue()
correction_queue = queue.Queue()
speech_text = ""
mic_active = False

# ✅ Finger Gesture History
history_length = 16
point_history = deque(maxlen=history_length)

# ✅ Sentence builder variables
sentence_text = ""
corrected_sentence = ""
last_sign = None
last_sign_time = 0
sign_add_delay = 1.5
last_correction_time = 0
correction_interval = 3.0

# ✅ Speech Detection Function
def detect_speech():
    global mic_active
    while True:
        time.sleep(3)
        mic_active = True
        print("🎤 Listening...")

        audio = sd.rec(int(sample_rate * recording_duration), samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()

        mic_active = False
        print("✅ Processing...")

        energy = np.sum(audio ** 2)

        if energy > energy_threshold:
            print(f"✅ Speech detected (Energy: {energy:.2f}). Sending to Distil-Whisper...")
            audio_queue.put(audio)
        else:
            print(f"❌ No speech detected (Energy: {energy:.2f}). Skipping.")

# ✅ Transcription Function
def transcribe_audio():
    global speech_text, sentence_text, last_correction_time
    while True:
        audio = audio_queue.get()
        if audio is None:
            break

        inputs = processor(audio[:, 0], sampling_rate=sample_rate, return_tensors="pt")
        inputs["attention_mask"] = torch.ones(inputs["input_features"].shape)
        outputs = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
        speech_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        print(f"📝 Transcribed: {speech_text}")
        sentence_text += " " + speech_text
        last_correction_time = time.time()

# ✅ Correction Thread (background Gemini correction)
def correction_worker():
    global corrected_sentence
    while True:
        text = correction_queue.get()
        if text is None:
            break
        corrected = correct_with_gemini(text)
        corrected_sentence = corrected

# ✅ Start Threads
speech_thread = threading.Thread(target=detect_speech, daemon=True)
whisper_thread = threading.Thread(target=transcribe_audio, daemon=True)
correction_thread = threading.Thread(target=correction_worker, daemon=True)
speech_thread.start()
whisper_thread.start()
correction_thread.start()

# ✅ Hand Preprocessing
def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0][0], temp[0][1]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    temp = list(np.array(temp).flatten().astype(np.float32))
    max_value = max(list(map(abs, temp)))
    if max_value > 0:
        temp = [x / max_value for x in temp]
    return temp

def pre_process_point_history(image, point_history):
    temp = copy.deepcopy(point_history)
    h, w = image.shape[0], image.shape[1]
    for i in range(len(temp)):
        temp[i][0] = temp[i][0] / w
        temp[i][1] = temp[i][1] / h
    return list(np.array(temp).flatten().astype(np.float32))

# ✅ Main Loop
while True:
    fps = fps_calc.get()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    debug_image = copy.deepcopy(frame)
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(image)

    sign_name = "Unknown"
    gesture_name = "None"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = [[int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])]
                             for landmark in hand_landmarks.landmark]

            pre_processed = pre_process_landmark(landmark_list)
            sign_id = keypoint_classifier(pre_processed)
            if 0 <= sign_id < len(keypoint_classifier_labels):
                sign_name = keypoint_classifier_labels[sign_id]

            sign_history.append(sign_name)

            if sign_id == 2:
                point_history.append(landmark_list[8])
            else:
                point_history.append([0, 0])

            pre_processed_point = pre_process_point_history(debug_image, point_history)
            if len(pre_processed_point) == (history_length * 2):
                gesture_id = point_history_classifier(pre_processed_point)
                gesture_name = point_history_classifier_labels[gesture_id]
                gesture_history.append(gesture_name)

    if len(sign_history) > 0:
        sign_name = Counter(sign_history).most_common(1)[0][0]

    if len(gesture_history) > 0:
        gesture_name = Counter(gesture_history).most_common(1)[0][0]

    # ✅ Build sentence logic
    current_time = time.time()
    if sign_name != "Unknown" and sign_name != last_sign:
        if current_time - last_sign_time > sign_add_delay:
            sentence_text += f" {sign_name}"
            last_sign = sign_name
            last_sign_time = current_time
            last_correction_time = current_time

    # ✅ Queue correction every few seconds (non-blocking)
    if time.time() - last_correction_time >= correction_interval and sentence_text.strip():
        correction_queue.put(sentence_text.strip())
        last_correction_time = time.time()

    # ✅ Draw UI
    mic_color = (0, 255, 0) if mic_active else (0, 0, 255)
    cv.circle(debug_image, (50, 200), 20, mic_color, -1)

    result_box_height = 150
    result_box = np.ones((result_box_height, debug_image.shape[1], 3), dtype=np.uint8) * 255

    cv.putText(result_box, f"Sign: {sign_name}", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 0), 2)
    cv.putText(result_box, f"Gesture: {gesture_name}", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 140, 0), 2)
    cv.putText(result_box, f"Speech: {speech_text}", (320, 45), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv.putText(result_box, f"Sentence: {sentence_text.strip()}", (20, 95), cv.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 2)
    cv.putText(result_box, f"Corrected: {corrected_sentence}", (20, 125), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 64, 255), 2)

    output_image = np.vstack((debug_image, result_box))
    cv.imshow("CareBuddy - Hand Gesture & Speech Recognition", output_image)

    key = cv.waitKey(10)
    if key == 27:  # ESC
        break
    elif key == ord('r'):
        sentence_text = ""
        corrected_sentence = ""
        print("🧹 Sentence cleared")

cap.release()
cv.destroyAllWindows()
audio_queue.put(None)
correction_queue.put(None)
