import cv2
import mediapipe as mp
from picamera2 import Picamera2
from gpiozero import LED
import serial
from time import sleep, time
import Comm
DEBUG = True
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils




# Hand detection status (shared between threads)
hand_in_roi = False

# Configuration: adjust as needed
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)




# Define detection rectangle (you can adjust these coordinates)
detection_rect = {
    'x': 200,      # Left edge
    'y': 150,      # Top edge
    'width': 240,  # Width
    'height': 180  # Height
}

def init_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    return picam2

picam2 = init_camera()
while True:

    message = Comm.receive_data()

    if not message or len(message) < 2:
        continue
    cmd = message[1]  
    match cmd:
            case 0x22:
                # Capture frame from Pi Camera
                frame = picam2.capture_array()

                # Convert to BGR format if needed (Picamera2 might output RGB or RGBA)
                if frame.shape[2] == 4:  # RGBA format
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] == 3 and frame.dtype == 'uint8':  # RGB format
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Flip image if you want mirror view
                # frame = cv2.flip(frame, 1)

                # Convert BGR?RGB for MediaPipe
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Let MediaPipe process and detect hands
                results = hands.process(img_rgb)

                # Draw the detection rectangle
                rect_color = (255, 0, 0)  # Blue rectangle by default
                hand_detected_in_rect = False

                # Draw detected hands & landmarks
                if results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks & connections
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                        # Check if any hand landmarks are inside the detection rectangle
                        for landmark in hand_landmarks.landmark:
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            
                            # Check if this landmark is inside the detection rectangle
                            if (detection_rect['x'] <= x <= detection_rect['x'] + detection_rect['width'] and
                                detection_rect['y'] <= y <= detection_rect['y'] + detection_rect['height']):
                                hand_detected_in_rect = True
                                break
                        
                        # Draw bounding box around hand (always draw this)
                        xs = [lm.x for lm in hand_landmarks.landmark]
                        ys = [lm.y for lm in hand_landmarks.landmark]
                        x_min = int(min(xs) * w)
                        x_max = int(max(xs) * w)
                        y_min = int(min(ys) * h)
                        y_max = int(max(ys) * h)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
                
                if hand_detected_in_rect:
                    rect_color = (0, 255, 0)  # Green when hand detected
                    print("Hand detected in rectangle!")

                # Draw the detection rectangle
                cv2.rectangle(frame, 
                            (detection_rect['x'], detection_rect['y']), 
                            (detection_rect['x'] + detection_rect['width'], 
                            detection_rect['y'] + detection_rect['height']), 
                            rect_color, 3)
                if DEBUG:
                    cv2.imshow("Hand Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                Comm.send_command(Comm.controller, [Comm.HandDetector_handCheck[0], int(hand_detected_in_rect)])
            case _ :
                print("Unknown command received")
    

# Cleanup
picam2.stop()
cv2.destroyAllWindows()