import cv2
import numpy as np
import tensorflow as tf

def preprocess(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    return gray

def extract_frame(img):
    frame = np.zeros(img.shape, np.uint8)

    thresh = cv2.adaptiveThreshold(img, 255, 0, 1, 9, 5)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    biggest_contour = []
    res = []
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if len(approx) == 4 and area > max_area and area > 40000:
            max_area = area
            biggest_contour = approx
    if len(biggest_contour) > 0:
        cv2.drawContours(frame, [biggest_contour], 0, 255, -1)
        cv2.drawContours(frame, [biggest_contour], 0, 0, 2)
        res = cv2.bitwise_and(img, frame)
    return res, biggest_contour, frame, thresh

# def get_corners(contour):
#     biggest_contour = contour.reshape(len(contour), 2)
#     sum_vectors = biggest_contour.sum(1)
#     sum_vectors2 = np.delete(biggest_contour, [np.argmax(sum_vectors), np.argmin(sum_vectors)], 0)

#     corners = np.float32([biggest_contour[np.argmin(sum_vectors)], sum_vectors2[np.argmax(sum_vectors2[:, 0])],
#                           sum_vectors2[np.argmin(sum_vectors2[:, 0])], biggest_contour[np.argmax(sum_vectors)]])

#     return corners

def get_corners(contour):
    biggest_contour = np.array(contour).reshape(len(contour), 2)
    sum_vectors = biggest_contour.sum(1)
    sum_vectors2 = np.delete(biggest_contour, [np.argmax(sum_vectors), np.argmin(sum_vectors)], 0)

    corners = np.float32([biggest_contour[np.argmin(sum_vectors)], sum_vectors2[np.argmax(sum_vectors2[:, 0])],
                          sum_vectors2[np.argmin(sum_vectors2[:, 0])], biggest_contour[np.argmax(sum_vectors)]])

    return corners

def perspective_transform(img, shape, corners):
    pts2 = np.float32(
        [[0, 0], [shape[0], 0], [0, shape[1]], [shape[0], shape[1]]])
    matrix = cv2.getPerspectiveTransform(corners, pts2)
    result = cv2.warpPerspective(img, matrix, (shape[0], shape[1]))
    return result

def extract_numbers(img):
    result = preprocess_numbers(img)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(result)
    viz = np.zeros_like(result, np.uint8)

    centroids_list = []
    stats_numbers = []

    for i, stat in enumerate(stats):
        if i == 0:
            continue
        if stat[4] > 50 and 5 <= stat[2] <= 40 and 5 <= stat[3] <= 40 and stat[0] > 0 and stat[1] > 0 and 1 <= int(
                stat[3] / stat[2]) <= 4:
            viz[labels == i] = 255
            centroids_list.append(centroids[i])
            stats_numbers.append(stat)

    stats_numbers = np.array(stats_numbers)
    centroids_list = np.array(centroids_list)
    return viz, stats_numbers, centroids_list

def preprocess_numbers(img):
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    return img

def center_numbers(img, stats, centroids):
    centered_num_grid = np.zeros_like(img, np.uint8)
    matrix_mask = np.zeros((9, 9), dtype='uint8')
    for i, number in enumerate(stats):
        left, top, width, height, area = stats[i]
        img_left = int(((left // 50)) * 50 + ((50 - width) / 2))
        img_top = int(((top // 50)) * 50 + ((50 - height) / 2))
        center = centroids[i]

        centered_num_grid[img_top:img_top + height,
        img_left: img_left + width] = img[number[1]:number[1] + number[3],
                                      number[0]:number[0] + number[2]]
        y = int(np.round((center[0] + 5) / 50, 1))
        x = int(np.round((center[1] + 5) / 50, 1))
        matrix_mask[x, y] = 1
    return centered_num_grid, matrix_mask

def process_cell(img):
    cropped_img = img[5:img.shape[0] - 5, 5:img.shape[0] - 5]
    resized = cv2.resize(cropped_img, (40, 40))
    return resized

def predict_numbers(numbers, matrix_mask, model):
    pred_list = []
    for row in range(9):
        for col in range(9):
            if matrix_mask[row, col] == 1:
                cell = numbers[50 * row: (50 * row) + 50, 50 * col: (50 * col) + 50]
                cell = process_cell(cell)
                cell = cell / 255
                cell = cell.reshape(1, 40, 40, 1)
                pred_list.append(cell)
    all_preds = model.predict(tf.reshape(np.array(pred_list), (np.sum(matrix_mask), 40, 40, 1)))
    proba = [np.max(pred) for pred in all_preds]
    preds = list(map(np.argmax, all_preds))
    flat_matrix = list(matrix_mask.flatten())

    i = 0
    for index, value in enumerate(flat_matrix):
        if value == 1:
            flat_matrix[index] = preds[i]
            i += 1

    flat_matrix = np.array(flat_matrix)
    matrix = flat_matrix.reshape(9, 9)
    return matrix

def displayNumbers(img, numbers, solved_num, color=(0, 255, 0)):
    w = int(img.shape[1] / 9)
    h = int(img.shape[0] / 9)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(9):
        for j in range(9):
            if numbers[j, i] == 0:
                cv2.putText(img, str(solved_num[j, i]),
                            (i * w + int(w / 2) - int((w / 4)), int((j + 0.7) * h)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, color,
                            1, cv2.LINE_AA)
    return img

def get_inv_perspective(img, masked_num, location, height=450, width=450):
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[1], location[2], location[3]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result

def draw_corners(img, corners):
    for corner in corners:
        x, y = corner
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    return img

def text_on_top(img, text1, color1, pos1, fps):
    cv2.rectangle(img, (0, 0), (1000, 40), (0, 0, 0), -1)
    cv2.putText(img=img, text=text1, org=pos1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                color=color1, thickness=1)
    cv2.putText(img=img, text=f'fps: {fps}', org=(35, 60),
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                color=(255, 255, 255), thickness=1)

    return img

def searching_rectangle(img, counter):
    corner_1 = (75 + (2 * counter), 75 + (2 * counter))
    corner_2 = (725 - (2 * counter), 525 - (2 * counter))
    cv2.rectangle(img, corner_1, corner_2, (0, 0, 255), 2)
    return img, corner_1[0]
