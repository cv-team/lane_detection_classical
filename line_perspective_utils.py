import cv2
import numpy as np
import matplotlib.pyplot as plt


def perspectiveChange(inputImg):
    # plt.imshow(inputImg)
    # plt.title("Inside Perspective Change")
    # plt.show()
    r, c = inputImg.shape[:2]
    source = np.float32([[c, r - 10],
                       [0, r - 10],
                       [680, 356],
                       [955, 349]])#[915, 356]])
    dest = np.float32([[c, r],
                     [0, r],
                     [600, 0],
                     [c - 20, 0]])

    perspTranform = cv2.getPerspectiveTransform(source, dest)

    invPerspTransform = cv2.getPerspectiveTransform(dest, source)

    warpedImage = cv2.warpPerspective(inputImg, perspTranform, (c, r), flags=cv2.INTER_LINEAR)
    return warpedImage, perspTranform, invPerspTransform

class Line:
    def __init__(self, length_buffer=10):
        self.pixel_last_iteration = None
        self.meter_last_iteration = None

        self.curvature = None
        self.x_coords = None
        self.y_coords = None

    def set_new_line(self, pixel_new, meter_new):
        self.meter_last_iteration = meter_new
        self.pixel_last_iteration = pixel_new

    def return_lane(self, img, lane_type):
        lane_width = 35
        height, width, color = img.shape
        y_plot = np.linspace(0, height - 1, height)
        coefficients = self.pixel_last_iteration
        central = coefficients[0] * y_plot ** 2 + coefficients[1] * y_plot + coefficients[2]
        left_lane = central - lane_width // 2
        right_lane = central + lane_width // 2
        left_coords = np.array(list(zip(left_lane, y_plot)))
        right_coords = np.array(np.flipud(list(zip(right_lane, y_plot))))
        final_coords = np.vstack([left_coords, right_coords])
        if lane_type == 'right':
            filled_area = cv2.fillPoly(img, [np.int32(final_coords)], (0, 0, 255))
        else:
            filled_area = cv2.fillPoly(img, [np.int32(final_coords)], (255, 0, 0))
        return filled_area

    @property
    def radius_of_curvature(self):
        coefficients = self.meter_last_iteration
        curvature = ((1 + pow(pow(coefficients[1], 2), 1.5)) / abs(2 * coefficients[0]))
        return curvature

def detect_lanes_from_binary(binary_img, left_line, right_line):
    # for i in range(binary_img.shape[0]):
    #     for j in range(0, 350):
    #         binary_img[i][j] = 0
    # for i in range(binary_img.shape[0]):
    #     for j in range(1500, binary_img.shape[1]):
    #         binary_img[i][j] = 0
    ht, wd = binary_img.shape
    hist = np.sum(binary_img[200:480, :], axis = 0)

    output_image = np.dstack((binary_img, binary_img, binary_img)) * 255
    midpt = len(hist) // 2
    start_point_left = np.argmax(hist[:midpt])
    start_point_right = np.argmax(hist[midpt:]) + midpt

    non_zero_points = binary_img.nonzero()
    y_coords = np.array(non_zero_points[0])
    x_coords = np.array(non_zero_points[1])

    currLeftCoord = start_point_left
    currRightCoord = start_point_right

    margin_width = 100
    recenterThresh = 50
    total_windows = 9
    singleWindowHt = int(ht / total_windows)

    leftLaneFinalCoords = []
    rightLaneFinalCoords = []
    for windowNum in range(total_windows):
        if windowNum < 3:
            continue
        y_window_down = ht - (windowNum + 1) * singleWindowHt
        y_window_up = ht - windowNum * singleWindowHt
        x_right_window_down = currRightCoord - margin_width
        x_right_window_up = currRightCoord + margin_width
        x_left_window_down = currLeftCoord - margin_width
        x_left_window_up = currLeftCoord + margin_width

        cv2.rectangle(output_image, (x_left_window_down, y_window_down), (x_left_window_up, y_window_up), (0, 255, 0), 2)
        cv2.rectangle(output_image, (x_right_window_down, y_window_down), (x_right_window_up, y_window_up), (0, 255, 0), 2)
        leftIndicesFound = ((y_coords >= y_window_down) & (x_coords < x_left_window_up) & (y_coords < y_window_up)
                                   & (x_coords >= x_left_window_down)).nonzero()[0]
        rightIndicesFound = ((y_coords >= y_window_down) & (x_coords < x_right_window_up) & (y_coords < y_window_up)
             & (x_coords >= x_right_window_down)).nonzero()[0]
        if len(rightIndicesFound) > recenterThresh:
            currRightCoord = np.int(np.mean(x_coords[rightIndicesFound]))
        if len(leftIndicesFound) > recenterThresh:
            currLeftCoord = np.int(np.mean(x_coords[leftIndicesFound]))

        leftLaneFinalCoords.append(leftIndicesFound)
        rightLaneFinalCoords.append(rightIndicesFound)
    plt.imshow(output_image)
    plt.title("Rectangles showing detected lanes")
    plt.show()
    rightLaneFinalCoords = np.hstack(rightLaneFinalCoords)
    leftLaneFinalCoords = np.hstack(leftLaneFinalCoords)
    rightLaneFinalCoords = rightLaneFinalCoords[::-1]
    leftLaneFinalCoords = leftLaneFinalCoords[::-1]

    left_line.x_coords = x_coords[leftLaneFinalCoords]
    left_line.y_coords = y_coords[leftLaneFinalCoords]
    right_line.x_coords = x_coords[rightLaneFinalCoords]
    right_line.y_coords = y_coords[rightLaneFinalCoords]

    pixel_new_left = np.polyfit(left_line.y_coords, left_line.x_coords, 2)
    pixel_new_right = np.polyfit(right_line.y_coords, right_line.x_coords, 2)

    meter_new_left = np.polyfit(left_line.y_coords * (30.0/720), left_line.x_coords * (3.7/700), 2)
    meter_new_right = np.polyfit(right_line.y_coords * (30.0/720), right_line.x_coords * (3.7/700), 2)

    left_line.set_new_line(pixel_new_left, meter_new_left)
    right_line.set_new_line(pixel_new_right, meter_new_right)

    output_image[y_coords[leftLaneFinalCoords], x_coords[leftLaneFinalCoords]] = [255, 0, 0]
    output_image[y_coords[rightLaneFinalCoords], x_coords[rightLaneFinalCoords]] = [0, 0, 255]

    return left_line, right_line, output_image