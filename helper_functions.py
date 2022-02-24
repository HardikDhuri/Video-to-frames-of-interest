import cv2
import os
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

def get_frames(path):
        dir_name = path.split("/")[-1].split(".")[0]
        dir_path = f"./{dir_name}"
        # Opens the inbuilt camera of laptop to capture video.
        cap = cv2.VideoCapture(path)
        current_frame = 0

        if os.path.exists(dir_path):
            print("Path already exists.\nPlease delete the existing folder and return get_frames again.")
            print(dir_path)
            return dir_name
        else:
            os.makedirs(dir_path)

            while(cap.isOpened()):
                    ret, frame = cap.read()

            # This condition prevents from infinite looping
            # incase video ends.
                    if ret == False:
                            break

                    # Save Frame by Frame into disk using imwrite method
                    cv2.imwrite(f'./{dir_name}/Frame{str(current_frame)}.jpg', frame)
                    current_frame += 1

            cap.release()
            cv2.destroyAllWindows()

        return dir_name

def prepare_image(path):
    img = np.array(load_img(path).resize((224,224)))
    return img

def display_batch(data_loader, class_names):
    (x, y) = next(data_loader)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        image, label = x[i], y[i]
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.title(class_names[label.argmax()])
        plt.axis("off")

def filter_frames(predictions, th = 0.8):
    class_names = ["electric car", "electric bus", "human"]
    filtered = {"index":[], "class":[]}
    for i, prediction in enumerate(predictions):
        class_ = class_names[prediction.argmax()]
        prob = prediction[prediction.argmax()]
        if (prob >= 0.3 and class_ == "human") or (class_ == "electric car" and prob >= th) or (class_ == "electric bus" and prob >= th):
            filtered["index"].append(i)
            filtered["class"].append(class_)
            if i % 500 == 0 and i != 0:
                start = filtered["index"][0]/24
                end = filtered["index"][-1]/24
                classes = list(set(filtered["class"]))
                print(f"From {start}s to {end}: {classes}")
                filtered = {"index": [],
                    "class": []}


# def load_images(dir_name):
#     image_set = [ prepare_image(image_name) for image_name in glob.glob(f"{dir_name}/*.jpg")]
#     return np.array(image_set)
