import os
import cv2

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


number_of_classes = 3
dataset_size = 800

# Iterate over possible camera indices to find a valid camera
camera_index = None
for i in range(5):  # Try up to 5 cameras
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        camera_index = i
        break
    else:
        cap.release()

# Check if a valid camera was found
if camera_index is None:
    print("Error: No valid camera found.")
    exit()

# Open the valid camera
cap = cv2.VideoCapture(camera_index)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    input("Press Enter when ready...")

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

        counter += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
