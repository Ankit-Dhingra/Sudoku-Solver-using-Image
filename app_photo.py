import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

warnings.filterwarnings("ignore", category=Warning, module="absl")

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.ERROR) 

import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image, ImageTk
from image_processing import perspective_transform, get_corners, draw_corners, text_on_top
from process import check_contour, predict, inv_transformation
import os


model = load_model('model3.h5')

# Function to handle image selection
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)

# Function to process the selected image
def process_image(image_path):
    # Load the image
    example = cv2.imread(image_path)

    # Check contour and preprocess the image
    contour_exist, preprocessed_img, frame, contour, contour_line, thresh = check_contour(example)

    # Get corners of the Sudoku grid
    corners = get_corners(contour)

    # Perform perspective transformation
    result = perspective_transform(frame, (450, 450), corners)

    # Predict numbers in Sudoku grid
    image_numbers, centered_numbers, predicted_matrix, solved_matrix, execution_time = predict(result, model)

    # Print the extracted puzzle
    print("Extracted Puzzle:")
    print(predicted_matrix)

    # Print the solved puzzle
    print("Solved Puzzle:")
    print(solved_matrix)

    # Create a mask for inverse transformation
    mask = np.zeros_like(result)

    # Perform inverse transformation
    example, solved_image = inv_transformation(mask, example, predicted_matrix, solved_matrix, corners)

    # Save the solved image to a temporary file
    temp_file_path = "solved_image.jpg"
    cv2.imwrite(temp_file_path, example)

    # Display the solved image in GUI
    display_images(image_path, temp_file_path)

    # Remove the temporary file
    os.remove(temp_file_path)

# Function to display the input and solved images in GUI
def display_images(input_image_path, solved_image_path):
    # Display input image
    input_img = Image.open(input_image_path)
    input_img = input_img.resize((450, 450), Image.LANCZOS)
    input_img = ImageTk.PhotoImage(input_img)
    input_image_label.config(image=input_img)
    input_image_label.image = input_img

    # Display solved image
    solved_img = Image.open(solved_image_path)
    solved_img = solved_img.resize((450, 450), Image.LANCZOS)
    solved_img = ImageTk.PhotoImage(solved_img)
    solved_image_label.config(image=solved_img)
    solved_image_label.image = solved_img

# GUI window
root = tk.Tk()
root.title("Sudoku Solver")

# Button to select image
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=10)

# Label to display the input image
input_image_label = tk.Label(root)
input_image_label.pack(side="left")

# Label to display the solved image
solved_image_label = tk.Label(root)
solved_image_label.pack(side="right")

root.mainloop()
