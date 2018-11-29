import cv2
import os
import matplotlib.pyplot as plt
from line_perspective_utils import perspectiveChange, detect_lanes_from_binary, Line
import numpy as np
import time
import imghdr

processed_frames = 0  # counter of frames processed (when processing video)
line_lt = Line()  # line on the left of the lane
line_rt = Line()  # line on the right of the lane


def process_pipeline(frame,test_img):
    global line_lt, line_rt, processed_frames

    height, width = frame.shape[:2]
    binIm = np.zeros(shape=(height, width), dtype=np.uint8)

    yHSVmin = np.array([0, 70, 70])
    yHSVmax = np.array([50, 255, 255])

    hueV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    minThreshold = np.all(hueV > yHSVmin, axis=2)
    maxThreshold = np.all(hueV < yHSVmax, axis=2)

    binIm = np.logical_or(binIm, np.logical_and(minThreshold, maxThreshold))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.equalizeHist(gray))
    plt.show()
    thre = 220
    _, eq_white_mask = cv2.threshold(cv2.equalizeHist(gray), thresh=thre, maxval=255, type=cv2.THRESH_BINARY)

    binIm = np.logical_or(binIm, eq_white_mask)

    kernel_size = 9
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobX = cv2.Sobel(grayScale, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobY = cv2.Sobel(grayScale, cv2.CV_64F, 0, 1, ksize=kernel_size)

    magnitudeS = np.sqrt(sobX ** 2 + sobY ** 2)
    magnitudeS = np.uint8(magnitudeS / np.max(magnitudeS) * 255)

    _, magnitudeS = cv2.threshold(magnitudeS, 50, 1, cv2.THRESH_BINARY)
    binIm = np.logical_or(binIm, magnitudeS.astype(bool))

    kernel = np.ones((5, 5), np.uint8)
    img_binary = cv2.morphologyEx(binIm.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    plt.imshow(img_binary)
    plt.title("Binary Image")
    plt.show()

    img_birdeye, M, Minv = perspectiveChange(img_binary)
    # plt.imshow(img_birdeye)
    # plt.show()

    line_lt, line_rt, img_fit = detect_lanes_from_binary(img_birdeye, line_lt, line_rt)

    # compute offset in meter from center of the lane
    line_lt_bottom = np.mean(line_lt.x_coords[line_lt.y_coords > 0.95 * line_lt.y_coords.max()])
    line_rt_bottom = np.mean(line_rt.x_coords[line_rt.y_coords > 0.95 * line_rt.y_coords.max()])
    lane_width = line_rt_bottom - line_lt_bottom
    midpoint = frame.shape[1] / 2
    offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
    offset_meter = (3.7 / 700) * offset_pix

    # draw the surface enclosed by lane lines back onto the original frame
    ht, wd, color = frame.shape
    warped = np.zeros_like(frame, dtype='uint8')
    unwarped = cv2.warpPerspective(warped, Minv, (wd, ht))
    onRoad = cv2.addWeighted(frame, 1., unwarped, 0.3, 0)

    line = np.zeros_like(frame)
    line = line_lt.return_lane(line, 'left')
    line = line_rt.return_lane(line, 'right')
    unwarped_line = cv2.warpPerspective(line, Minv, (wd, ht))
    maskForLine = onRoad.copy()
    indices = np.any([unwarped_line != 0][0], axis=2)
    coordPoints = np.where(indices == True)
    fileName = test_img[:-4] + "_output_thresh_" + str(thre) + ".txt"
    f = open(fileName, "w+")
    writeStr = ""
    writeStr2 = ""
    prevY = 0
    prevYY = 0
    for y, x in zip(np.ndarray.tolist(coordPoints[0]), np.ndarray.tolist(coordPoints[1])):

        if x < 800:
            if y != prevY:
                writeStr += str(x) + " " + str(y) + " "
            prevY = y
        else:
            if y != prevYY:
                writeStr2 += str(x) + " " + str(y) + " "
            prevYY = y

    f.write(writeStr + "\n")
    f.write(writeStr2)
    f.close()
    maskForLine[indices] = unwarped_line[indices]
    blend_on_road = cv2.addWeighted(src1=maskForLine, alpha=0.8, src2=onRoad, beta=0.5, gamma=0.)
    # blend_on_road = draw_on_road(frame, Minv, line_lt, line_rt)

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([line_lt.radius_of_curvature, line_rt.radius_of_curvature])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (860, 130), font, 0.9,
                (0, 0, 0), 2, cv2.LINE_AA)
    ##

    processed_frames += 1

    return blend_on_road, fileName, thre


if __name__ == '__main__':
    # test_img_dir = 'Dark/06042013_0512.MP4/'
    # test_img_dir = 'test_images/'
    test_img_dir = 'crowd_test/'
    num_images = 0
    mean = 0
    for test_img in os.listdir(test_img_dir):
        # if (imghdr.what(test_img_dir + test_img)) == 'jpeg':
        if len(test_img) == 9:
            frame = cv2.imread(os.path.join(test_img_dir, test_img))
            start_time = int(round(time.time() * 1000))
            blend, fileName, thre = process_pipeline(frame,test_img)
            end_time = int(round(time.time() * 1000))
            mean += end_time - start_time
            num_images += 1
            cv2.imwrite('output_images/{}'.format(test_img), blend)
            plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
            plt.title("Final Image with Lanes For Threhold = " + str(thre))
            plt.savefig(fileName[:-4] + '.jpg')
            plt.show()
    mean /= num_images
    print(mean)