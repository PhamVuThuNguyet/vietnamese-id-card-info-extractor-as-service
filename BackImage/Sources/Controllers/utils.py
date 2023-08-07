import cv2
import numpy as np


def class_order(boxes, categories):
    Z = []
    ctg = np.argsort(categories)

    for index in ctg:
        Z.append(boxes[index])

    return Z


def get_center_point(box):
    x1, y1, x2, y2 = box

    return x1 + ((x2 - x1) // 2), y1 + ((y2 - y1) // 2)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    image = np.asarray(image)

    rect = order_points(pts)

    (top_left, top_right, bottom_right, bottom_left) = rect

    width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))

    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped

def non_maximum_suppression(boxes, labels, overlapThresh):
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # If the bounding boxes are integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]

    # Compute the area of the bounding boxes and sort them by the bottom-right y-coordinate
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while the list is not empty
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the value to picked list
        last_index = len(idxs) - 1
        i = idxs[last_index]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start
        # Find the smallest (x, y) coordinates for the end
        start_x = np.maximum(x1[i], x1[idxs[:last_index]])
        start_y = np.maximum(y1[i], y1[idxs[:last_index]])
        end_x = np.minimum(x2[i], x2[idxs[:last_index]])
        end_y = np.minimum(y2[i], y2[idxs[:last_index]])

        # Compute the width and height of the bounding box
        width = np.maximum(0, end_x - start_x + 1)
        height = np.maximum(0, end_y - start_y + 1)

        # Compute the ratio of overlap
        overlap = (width * height) / area[idxs[:last_index]]

        # Delete all overlap boxes
        idxs = np.delete(idxs, np.concatenate(([last_index], np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked using the integer data tupe
    final_labels = [labels[idx] for idx in pick]
    final_boxes = boxes[pick].astype("int")

    return final_boxes, final_labels
