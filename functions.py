import cv2
import os
import numpy as np

def load_image(file_path):
    return cv2.imread(file_path)

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY_INV)
    return np.array(thresh)

def load_digits():
    folder_path = r'Digits/'
    categories = [str(number) for number in range(10)]
    data = []
    
    for category in categories:
        category_folder = os.path.join(folder_path, category)
        for img in os.listdir(category_folder):
            img_path = os.path.join(category_folder, img)
            label = int(categories.index(category))
            arr = cv2.imread(img_path)
            new_arr = cv2.resize(arr, (40, 40))
            new_arr = preprocess_image(new_arr)
            data.append([new_arr, label])
    return data

def load_sudoku_images(folder):
    data = []
    
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        arr = cv2.imread(img_path)
        new_arr = cv2.resize(arr, (540, 540), interpolation=cv2.INTER_LINEAR)
        data.append(new_arr)
    
    data = np.array(data)
    return data
