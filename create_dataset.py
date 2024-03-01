import os
import pickle
import cv2
import mediapipe as mp

def load_image(file_path):
    try:
        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def process_landmarks(results):
    data_aux = []

    for hand_landmarks in results.multi_hand_landmarks:
        x_ = [landmark.x for landmark in hand_landmarks.landmark]
        y_ = [landmark.y for landmark in hand_landmarks.landmark]

        min_x, min_y = min(x_), min(y_)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            data_aux.append(x - min_x)
            data_aux.append(y - min_y)

    return data_aux

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        file_path = os.path.join(DATA_DIR, dir_, img_path)
        img_rgb = load_image(file_path)

        if img_rgb is not None:
            print(f"Processing image: {file_path}")

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                data_aux = process_landmarks(results)
                data.append(data_aux)
                labels.append(dir_)

# Save the data to a pickle file
output_file_path = 'data.pickle'
with open(output_file_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

hands.close()
print(f"Dataset successfully created and saved to {output_file_path}.")
