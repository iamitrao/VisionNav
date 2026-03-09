import cv2
import time
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
import pyautogui
import os

# Adjust PyAutoGUI preferences
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

print("Loading Models...")
try:
    model = joblib.load('blink_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Loaded model and scaler successfully.")
except Exception as e:
    print("Error loading models:", e)
    exit(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Feature extraction function exactly matching notebook 02
def extract_single_frame_features(image, results):
    if not results.multi_face_landmarks: return None
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape
    pts = [(int(pt.x * w), int(pt.y * h)) for pt in landmarks]
    
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    l_p0, l_p1, l_p2, l_p3, l_p4, l_p5 = [pts[i] for i in LEFT_EYE]
    r_p0, r_p1, r_p2, r_p3, r_p4, r_p5 = [pts[i] for i in RIGHT_EYE]
    
    lear = (euclidean_distance(l_p1, l_p5) + euclidean_distance(l_p2, l_p4)) / (2.0 * euclidean_distance(l_p0, l_p3))
    rear = (euclidean_distance(r_p1, r_p5) + euclidean_distance(r_p2, r_p4)) / (2.0 * euclidean_distance(r_p0, r_p3))
    avg_ear = (lear + rear) / 2.0
    
    l_w = euclidean_distance(l_p0, l_p3)
    l_h = (euclidean_distance(l_p1, l_p5) + euclidean_distance(l_p2, l_p4)) / 2.0
    r_w = euclidean_distance(r_p0, r_p3)
    r_h = (euclidean_distance(r_p1, r_p5) + euclidean_distance(r_p2, r_p4)) / 2.0
    
    l_op = l_h / (l_w + 1e-6)
    r_op = r_h / (r_w + 1e-6)
    
    pv = (l_h + r_h) / 2.0 
    
    l_eb = pts[105]
    r_eb = pts[334]
    l_ed = euclidean_distance(l_p1, l_eb)
    r_ed = euclidean_distance(r_p1, r_eb)
    
    features = [lear, rear, avg_ear, l_w, l_h, r_w, r_h, l_op, r_op, l_h, r_h, pv, l_ed, r_ed]
    return np.array(features).reshape(1, -1), avg_ear

cap = cv2.VideoCapture(0)

# State Variables
last_state = 0  # 0: Open, 1: Closed
blink_timestamps = []

# Application Logic State
launcher_open = False
last_action_time = time.time()
cooldown = 0.5  # Reduced cooldown to make it feel more responsive

# Hand Tracking State
last_hand_pos = None
swipe_threshold = 0.1  # percentage of screen width/height hand needs to move to trigger a swipe

print("=========================================")
print("      OS BLINK & HAND CONTROLLER         ")
print("=========================================")
print("  Double Blink: Open Applications Folder")
print("  Swipe Hand: Left/Right/Up/Down to Navigate")
print("  Double Blink (while open): Open Selected App")
print("=========================================")
print("Press 'Q' inside the camera window to stop.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip frame horizontally for intuitive hand movement
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process both Face Mesh and Hands
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    
    pred_label = "Waiting..."
    active_command = "None"
    
    current_time = time.time()

    # 1. PROCESS HANDS FOR NAVIGATION
    if hand_results.multi_hand_landmarks and launcher_open:
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        # Use index finger tip (8) for position
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = index_tip.x, index_tip.y
        
        # Draw hand landmarks
        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Swipe tracking logic
        if last_hand_pos is None:
            last_hand_pos = (x, y)
        else:
            dx = x - last_hand_pos[0]
            dy = y - last_hand_pos[1]
            
            # Check for swipe if cooldown has passed
            if current_time - last_action_time > cooldown:
                moved = False
                if dx < -swipe_threshold:
                    print("Swipe Left! -> Moving Left")
                    pyautogui.press('left')
                    active_command = "Swipe: Left"
                    moved = True
                elif dx > swipe_threshold:
                    print("Swipe Right! -> Moving Right")
                    pyautogui.press('right')
                    active_command = "Swipe: Right"
                    moved = True
                elif dy < -swipe_threshold:
                    print("Swipe Up! -> Moving Up")
                    pyautogui.press('up')
                    active_command = "Swipe: Up"
                    moved = True
                elif dy > swipe_threshold:
                    print("Swipe Down! -> Moving Down")
                    pyautogui.press('down')
                    active_command = "Swipe: Down"
                    moved = True
                
                if moved:
                    last_action_time = current_time
                    last_hand_pos = (x, y) 
                    
        # Visual anchor for the user
        h, w, _ = frame.shape
        anchor_x = int(last_hand_pos[0] * w) if last_hand_pos else int(w/2)
        anchor_y = int(last_hand_pos[1] * h) if last_hand_pos else int(h/2)
        cv2.circle(frame, (anchor_x, anchor_y), 10, (0, 0, 255), -1)
        
        # Draw movement borders around anchor
        box_w = int(w * swipe_threshold)
        box_h = int(h * swipe_threshold)
        cv2.rectangle(frame, (anchor_x - box_w, anchor_y - box_h), (anchor_x + box_w, anchor_y + box_h), (255, 0, 0), 2)
        
    else:
        # Reset anchor when hand disappears
        last_hand_pos = None

    # 2. PROCESS FACE FOR BLINKS
    if face_results.multi_face_landmarks:
        feat_data = extract_single_frame_features(frame, face_results)
        if feat_data is not None:
            features, ear = feat_data
            
            scaled_feat = scaler.transform(features)
            pred_class = model.predict(scaled_feat)[0]
            pred_label = "CLOSED" if pred_class == 1 else "OPEN"
            
            # Blink Detection Logic
            if last_state == 1 and pred_class == 0:  # Eye just opened
                if len(blink_timestamps) > 0 and (current_time - blink_timestamps[-1]) < 0.6:
                    # ---- DOUBLE BLINK DETECTED ----
                    if current_time - last_action_time > cooldown:
                        if not launcher_open:
                            print("Double Blink! -> Opening Applications Folder")
                            os.system('open /Applications')
                            launcher_open = True
                            active_command = "Double Blink: Opened Applications"
                        else:
                            print("Double Blink! -> Opening Selected App (Cmd+Down)")
                            pyautogui.keyDown('command')
                            pyautogui.press('down')
                            pyautogui.keyUp('command')
                            launcher_open = False
                            active_command = "Double Blink: Opened App"
                        last_action_time = current_time
                        
                    blink_timestamps = [] # Reset
                else:
                    # ---- WAIT FOR POTENTIAL DOUBLE BLINK ----
                    blink_timestamps.append(current_time)
            
            last_state = pred_class

    # Overlay Text for debugging
    h, w, _ = frame.shape
    cv2.putText(frame, f"Eye State: {pred_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Folder Open: {launcher_open}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Last Action: {active_command}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "To Move: Swipe finger OUTSIDE the blue box", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Blink & Hand Controller", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Controller deactivated.")
