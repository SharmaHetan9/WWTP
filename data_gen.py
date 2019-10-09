import numpy as np
import os, random, cv2
import matplotlib.pyplot as plt

PATH = os.getcwd()+"\\photos"
CATEGORIES = os.listdir(PATH)
IMG_SIZE = 75


def create_data():
    data = []

    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        path = os.path.join(PATH, category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                data.append([new_array, class_num])

            except Exception as e:
                print(e)

    random.shuffle(data)
                
    X, y = [], []

    for feature, label in data:
        X.append(feature)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    np.save("X.npy", X)
    np.save("y.npy", y)

create_data()
