import cv2
import numpy as np
import time
import object_socket

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')

width = 380
height = 260
upper_left = (int(width * 0.42), int(height * 0.778))
upper_right = (int(width * 0.55), int(height * 0.778))
lower_left = (width * 0.045, int(height))
lower_right = (width * 0.985, int(height))

s = object_socket.ObjectReceiverSocket('127.0.0.1', 5000, print_when_connecting_to_sender=True, print_when_receiving_object=True)

while True:

    start = time.perf_counter()

    #ret, frame = cam.read()
    ret, frame = s.recv_object()

    if ret is False:
        break

    # ORIGINAL (1)
    cv2.imshow('Original', frame)

    # resized frame and print (2)
    resized = cv2.resize(frame, (width, height))
    cv2.imshow('Resized', resized)

    # grayscale frame and print (3)
    grayScaled = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscaled', grayScaled)

    # select only the road in trapezoid shape (4)
    onlyRoad = np.zeros((height, width), dtype = np.uint8)
    trapezoid = np.array([upper_right, upper_left, lower_left, lower_right], dtype = np.int32)
    cv2.fillConvexPoly(onlyRoad, trapezoid, 1)
    cv2.imshow("Trapezoid", onlyRoad * 255)

    Trapezoid = onlyRoad * grayScaled
    cv2.imshow('Only the Road in trapezoid shape', Trapezoid)

    # stretch the road (5)
    trapezoid_bounds = np.float32([upper_right, upper_left, lower_left, lower_right])
    onlyRoad = np.float32([[width, 0], [0, 0], [0, height], [width, height]])
    magicalMatrix = cv2.getPerspectiveTransform(trapezoid_bounds, onlyRoad)
    roadStretched = cv2.warpPerspective(Trapezoid, magicalMatrix, (width, height))
    cv2.imshow('The Road Stretched', roadStretched)

    # Add a bit of blur (6)
    roadStretchedBlurred = cv2.blur(roadStretched, ksize = (9,9))
    cv2.imshow('The Road Stretched with BLUR', roadStretchedBlurred)

    # Do edge detection (7)
    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])
    sobel_horizontal = np.transpose(sobel_vertical)

    roadStretchedBlurredfloat32 = np.float32(roadStretchedBlurred)
    filtImgVertical = cv2.filter2D(roadStretchedBlurredfloat32, -1, sobel_vertical)
    filtImgHorizontal = cv2.filter2D(roadStretchedBlurredfloat32, -1, sobel_horizontal)
    resFilterredImg = np.sqrt(filtImgVertical**2 + filtImgHorizontal**2)
    resultingImg = cv2.convertScaleAbs(resFilterredImg)
    cv2.imshow('The Edges Detected', resultingImg)

    # Binarize the frame (8)
    treshholdValue = int(255/12)
    ret1,thresh1 = cv2.threshold(resultingImg, treshholdValue,255,cv2.THRESH_BINARY)
    cv2.imshow('The Edges Detected BINARIZED', thresh1)

    # Get the coordinates of street markings on each side of the road (9)
    copiedImage = thresh1.copy()
    copiedImage[:, 0:int(width * 0.05)] = 0 #left image cropped
    copiedImage[:, int(width * 0.97):] = 0 #right image cropped
    copiedImage[int(height * 0.97):, :] = 0 #bottom image cropped
    cv2.imshow('The Edges Detected BINARIZED & CROPPED', copiedImage)

    copiedImageLeft = copiedImage[:, 0:int(width * 0.50)]
    copiedImageRight = copiedImage[:, int(width * 0.50):]
    cv2.imshow('L', copiedImageLeft)
    cv2.imshow('R', copiedImageRight)

    leftHalf_values = np.argwhere(copiedImageLeft == 255)
    rightHalf_values = np.argwhere(copiedImageRight == 255)
    rightHalf_values[:, 1] += int(width / 2)

    left_xs = leftHalf_values[:, 1]
    left_ys = leftHalf_values[:, 0]

    right_xs = rightHalf_values[:, 1]
    right_ys = rightHalf_values[:, 0]

    # Find the lines that detect the edges of the lane (10)
    left_regression = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg = 1)
    right_regression = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg = 1)

    left_top_y = 0
    if left_regression[1] is not 0 or  -left_regression[0] / left_regression[1] in range (-10**8, 10**8):
        left_top_x = (left_top_y - left_regression[0]) / left_regression[1]

    left_bottom_y = height
    if left_regression[1] is not 0 or (height - left_regression[0]) / left_regression[1] in range (-10**8, 10**8):
        left_bottom_x = int((height - left_regression[0]) / left_regression[1])

    left_bottom_x = (left_bottom_y - left_regression[0]) / left_regression[1]

    right_top_y = 0
    if right_regression[1] is not 0 or right_regression[0] / right_regression[1] in range (-10**8, 10**8):
        right_top_x = (right_top_y - right_regression[0]) / right_regression[1]

    right_bottom_y = height
    if right_regression[1] is not 0 or (height - right_regression[0]) / right_regression[1] in range (-10**8, 10**8):
        right_bottom_x = (right_bottom_y - right_regression[0]) / right_regression[1]

    left_top = int(left_top_x), left_top_y
    left_bottom = int(left_bottom_x), left_bottom_y

    right_top = int(right_top_x), right_top_y
    right_bottom = int(right_bottom_x), right_bottom_y

    cv2.line(copiedImage, left_top, left_bottom, (200, 0, 0), 5)
    cv2.line(copiedImage, right_top, right_bottom, (100, 0, 0), 5)
    cv2.imshow('Lines on Image', copiedImage)

    #Create a final visualization! (11)

#left_side_frame
    #a
    finalResultLeft = np.zeros((height, width), dtype = np.uint8)
    #b
    left_side_frame = cv2.line(finalResultLeft, left_top, left_bottom, (255, 0, 0), 3)
    #c
    stretcheed = cv2.getPerspectiveTransform(np.array([(width, 0), (0, 0), (0, height), (width, height)], dtype = np.float32), trapezoid_bounds)
    #d
    left_side_frame = cv2.warpPerspective(left_side_frame, stretcheed, (width, height))
    #e
    left_side_frame_Copied = np.array(left_side_frame, copy=True)
    left_points = np.argwhere(left_side_frame_Copied > 0)
    left_xs = np.array(left_points[:, 1])
    left_ys = np.array(left_points[:, 0])

#right_side_frame
    # a
    finalResultRight = np.zeros((height, width), dtype=np.uint8)
    # b
    right_side_frame = cv2.line(finalResultRight, right_top, right_bottom, (255, 0, 0), 3)
    # c
    stretcheed = cv2.getPerspectiveTransform(np.array([(width, 0), (0, 0), (0, height), (width, height)], dtype=np.float32), trapezoid_bounds)
    # d
    right_side_frame = cv2.warpPerspective(right_side_frame, stretcheed, (width, height))
    # e
    right_side_frame_Copied = np.array(right_side_frame, copy=True)
    right_points = np.argwhere(right_side_frame_Copied > 0)
    right_xs = np.array([x + width // 2 for x in right_points[:, 1]])
    right_ys = np.array(right_points[:, 0])

    #cv2.imshow('Final', left_side_frame_Copied)
    #cv2.imshow('Final1', right_side_frame_Copied)

    finalResult = resized.copy()
    for i in range(len(left_points)):
        finalResult[left_points[i][0], left_points[i][1]] = (50, 50, 250)
    for i in range(len(right_points)):
        finalResult[right_points[i][0], right_points[i][1]] = (50, 250, 50)
    finalResult = cv2.resize(finalResult, (resized.shape[1], resized.shape[0]))
    cv2.imshow("Finally", finalResult)

    stop = time.perf_counter()
    print(1 // (stop-start))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
s.close()