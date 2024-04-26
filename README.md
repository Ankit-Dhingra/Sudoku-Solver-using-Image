# Sudoku Solver

This project is a Sudoku solver that uses image recognition to extract puzzles from images and solve them. It is implemented in Python and utilizes various libraries such as OpenCV, TensorFlow, and tkinter for GUI.

## Overview

The Sudoku Solver allows users to upload an image containing a Sudoku puzzle, processes the image to extract the puzzle, solves the puzzle, and displays the solved puzzle along with the input image. The solution is obtained through image processing techniques and deep learning-based digit recognition.

## Features

- **Image Upload:** Users can upload an image containing a Sudoku puzzle through the GUI.
- **Image Processing:** The uploaded image is preprocessed to extract the Sudoku grid using contour detection and perspective transformation.
- **Digit Recognition:** The digits in the Sudoku grid are recognized using a pre-trained deep learning model.
- **Sudoku Solving:** The recognized digits are used to solve the Sudoku puzzle.
- **GUI Display:** The input image and the solved puzzle are displayed in the graphical user interface.

## Used Tools and Libraries

The Sudoku Solver project utilizes the following tools and libraries:

- **Python:** The primary programming language used for developing the application.
- **OpenCV:** A computer vision library used for image processing tasks such as contour detection and perspective transformation.
- **TensorFlow:** An open-source machine learning framework used for building and training deep learning models.
- **Keras:** A high-level neural networks API, used here for loading and using pre-trained deep learning models.
- **tkinter:** The standard GUI toolkit for Python, used for building the graphical user interface of the application.
- **NumPy:** A fundamental package for scientific computing with Python, used for numerical operations and array manipulation.

These tools and libraries were instrumental in implementing various functionalities of the Sudoku Solver, including image processing, deep learning-based digit recognition, and GUI development.

## Example Results

### GUI Output
![GUI Output](https://github.com/Ankit-Dhingra/Sudoku-Solver-using-Image/raw/b7281fa6a743e7e3abfb0de83b14acdfddc8b4ab/Screenshot%202024-03-10%20203414.png)

### Extracted Puzzle
![Extracted Puzzle](https://github.com/Ankit-Dhingra/Sudoku-Solver-using-Image/raw/b7281fa6a743e7e3abfb0de83b14acdfddc8b4ab/Screenshot%202024-04-21%20011334.png)

### Solved Puzzle
![Solved Puzzle](https://github.com/Ankit-Dhingra/Sudoku-Solver-using-Image/raw/b7281fa6a743e7e3abfb0de83b14acdfddc8b4ab/Screenshot%202024-04-21%20011353.png)


## Installation

1. Clone the repository:
git clone https://github.com/Ankit-Dhingra/sudoku-solver.git


2. Install the required dependencies:


3. Run the application:

## Usage

1. Launch the application by running `app_photo.py`.
2. Click on the "Select Image" button to choose an image containing a Sudoku puzzle.
3. Once the image is selected, the application processes it, solves the puzzle, and displays the input image along with the solved puzzle.
