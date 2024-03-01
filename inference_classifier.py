import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Define labels_dict before the loop
labels_dict = {0: 'hello', 1: 'i love you', 2: 'bon'}

while True:
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error capturing frame")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks using MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

            # Extract hand landmarks for prediction
            data_aux = []
            for i in range(21):  # Use all 21 landmarks
                data_aux.extend([hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y])

            # Predict gesture using all features
            prediction = model.predict([np.asarray(data_aux)])

            # Display predicted gesture and bounding box
            x1, y1 = int(min(hand_landmarks.landmark[1].x * W, hand_landmarks.landmark[5].x * W)) - 10, int(
                min(hand_landmarks.landmark[1].y * H, hand_landmarks.landmark[9].y * H)) - 10
            x2, y2 = int(max(hand_landmarks.landmark[5].x * W, hand_landmarks.landmark[9].x * W)) + 10, int(
                max(hand_landmarks.landmark[13].y * H, hand_landmarks.landmark[17].y * H)) + 10

            predicted_label = labels_dict[int(prediction[0])]
            true_label_of_current_frame = 0  # Replace with the true label of the current frame

            # Print predicted and true labels
            #print("Predicted Label:", predicted_label)
            #print("True Label:", labels_dict[true_label_of_current_frame])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
