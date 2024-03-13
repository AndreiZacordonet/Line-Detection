import cv2
import numpy as np

import object_socket


def road_selection(frame):
    """ Selects the road in a trapezoid shape """
    height, width = frame.shape[0], frame.shape[1]
    upper_right = int(width * 0.55), int(height * 0.75)  # int(width*0.56), int(height*0.74) | => cu astea mai ramane
    upper_left = int(width * 0.45), int(height * 0.75)  # int(width*0.44), int(height*0.74) |  putin din marginea de sus
    lower_left = 0, height
    lower_right = width, height
    trap = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)
    trap_img = np.zeros((height, width), dtype=np.uint8)
    result_frame = cv2.fillConvexPoly(trap_img, trap, 1) * frame
    return result_frame, trap


def stretching(frame, trapezoid):
    trap = np.float32(trapezoid)
    width, height = frame.shape[1], frame.shape[0]
    stretched = cv2.getPerspectiveTransform(trap, np.array([(width, 0), (0, 0), (0, height), (width, height)],
                                                           dtype=np.float32))
    result_frame = cv2.warpPerspective(frame, stretched, (width, height))
    return result_frame


def edge_detection(frame):
    sobel_vertical = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_horizontal = np.transpose(sobel_vertical)
    filter1 = cv2.filter2D(np.float32(frame), -1, sobel_vertical)
    filter2 = cv2.filter2D(np.float32(frame), -1, sobel_horizontal)
    # cv2.imshow("vertical sobel", cv2.convertScaleAbs(filter1))    # conversion from float32 in uint8
    # cv2.imshow("horizontal sobel", cv2.convertScaleAbs(filter2))
    result_frame = cv2.convertScaleAbs(np.sqrt(filter1 ** 2 + filter2 ** 2))
    return result_frame


def white_lines_coord(frame):
    edge_corected = np.array(frame, copy=True)
    height, width = frame.shape[0], frame.shape[1]
    edge_corected[0:height, 0:int(width * 0.1)] = 0  #
    edge_corected[0:height, int(width * 0.9):width] = 0  # coloring the column margins black
    half1 = np.array(edge_corected[0:height, 0:width // 2])  #
    half2 = np.array(edge_corected[0:height, width // 2:width + 1])  #
    left_points = np.argwhere(half1 > 0)  #
    right_points = np.argwhere(half2 > 0)  # selecting white point coordinates
    return left_points, right_points

def pinguu():
    # se comenteaza
    cam = cv2.VideoCapture(
        r'C:\Users\andre\OneDrive\Desktop\Materii\Anul_III\Semestrul_I\ISSA\Lab3\Lane Detection Test Video-01.mp4')

    # se decomenteaza
    # s = object_socket.ObjectReceiverSocket('127.0.0.1', 5000, print_when_connecting_to_sender=True, print_when_receiving_object=True)
    left_top_x = 0
    left_bottom_x = 0
    right_top_x = 0
    right_bottom_x = 0

    while True:
        # se comenteaza
        ret, frame = cam.read()

        # ret, frame = s.recv_object() se decomenteaza

        if not ret:
            break

        # resize window
        height, width = int(frame.shape[0] / 2.5), int(frame.shape[1] / 4.5)
        frame_resized = cv2.resize(frame, (width, height))

        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)    # gray image

        frame_road, trapezoid = road_selection(frame_gray)  # only_road selection:

        frame_stretched = stretching(frame_road, trapezoid)  # trapezoid stretching

        frame_blured = cv2.blur(frame_stretched, ksize=(9, 9))      # blur effect, matrix dimensions => (9, 9)

        frame_processed = edge_detection(frame_blured)  # edge detection with sobel filters

        # binarize the image
        frame_binarized = cv2.threshold(frame_processed, 60, 255, cv2.THRESH_BINARY)[1]      # try with THRESH_OTSU
        # cv2.imshow("OTSU", cv2.threshold(frame_processed, 128, 255, cv2.THRESH_OTSU)[1])

        # street coordinates
        left_points, right_points = white_lines_coord(frame_binarized)
        if len(right_points) == 0:
            print("pipi")
            continue
        left_xs = np.array(left_points[:, 1])
        left_ys = np.array(left_points[:, 0])
        right_xs = np.array([x+width//2 for x in right_points[:, 1]])
        right_ys = np.array(right_points[:, 0])

        # linear regression
        left_equation = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1, rcond=1e-10)   # deg = degree (one)    | (a, b)
        right_equation = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1, rcond=1e-8)   # deg = degree (one) | y = ax + b
        left_top_y = 0
        if left_equation[1] is not 0 or -left_equation[0]/left_equation[1] in range(-10**8, 10**8):
            left_top_x = int(-left_equation[0] / left_equation[1])
        left_bottom_y = height
        if left_equation[1] is not 0 or (height - left_equation[0]) / left_equation[1] in range(-10 ** 8, 10 ** 8):
            left_bottom_x = int((height - left_equation[0]) / left_equation[1])
        right_top_y = 0
        if right_equation[1] is not 0 or -right_equation[0] / right_equation[1] in range(-10 ** 8, 10 ** 8):
            right_top_x = int(-right_equation[0] / right_equation[1])
        right_bottom_y = height
        if right_equation[1] is not 0 or (height - right_equation[0]) / right_equation[1] in range(-10 ** 8, 10 ** 8):
            right_bottom_x = int((height - right_equation[0]) / right_equation[1])

        left_top = left_top_x, left_top_y
        left_bottom = left_bottom_x, left_bottom_y
        right_top = right_top_x, right_top_y
        right_bottom = right_bottom_x, right_bottom_y

        frame_lined = cv2.line(frame_binarized, left_top, left_bottom, (128, 128, 128), 3)
        try:
            frame_lined = cv2.line(frame_lined, right_top, right_bottom, (128, 128, 128), 3)
        except cv2.error:
            print("cv2.error")
            pass
        cv2.imshow("left", frame_lined)

        # final visualization
        blank_frame = np.zeros((height, width), dtype=np.uint8)
        left_lined_frame = cv2.line(blank_frame, left_top, left_bottom, (255, 0, 0), 3)
        stretched = cv2.getPerspectiveTransform(np.array([(width, 0), (0, 0), (0, height), (width, height)], dtype=np.float32), trapezoid)
        left_streached_frame = cv2.warpPerspective(left_lined_frame, stretched, (width, height))
        cv2.imshow("left streched frame", left_streached_frame)
        right_lined_frame = cv2.line(blank_frame, right_top, right_bottom, (255, 0, 0), 3)
        right_streached_frame = cv2.warpPerspective(right_lined_frame, stretched, (width, height))
        cv2.imshow("right streched frame", right_streached_frame)

        edge_corected = np.array(right_streached_frame, copy=True)
        half1 = np.array(edge_corected[0:height, 0:width // 2])  #
        half2 = np.array(edge_corected[0:height, width // 2:width])  #
        left_points = np.argwhere(half1 > 0)  #
        right_points = np.argwhere(half2 > 0)  # selecting white point coordinates

        # left_xs = np.array(left_points[:, 1])
        # left_ys = np.array(left_points[:, 0])q
        # right_xs = np.array([x + width // 2 for x in right_points[:, 1]])
        # right_ys = np.array(right_points[:, 0])

        frame_copy = frame_resized.copy()
        for i in range(len(left_points)):
            frame_copy[left_points[i][0], left_points[i][1]] = (50, 50, 250)
        for i in range(len(right_points)):
            frame_copy[right_points[i][0], right_points[i][1] + width//2] = (50, 250, 50)
        frame_copy = cv2.resize(frame_copy, (width, height)) # (frame.shape[1], frame.shape[0]))
        cv2.imshow("Finally", frame_copy)


        # displaying results
        cv2.imshow('Gray', frame_gray)
        cv2.imshow("Only Road", frame_road)
        cv2.imshow("Stretched image", frame_stretched)
        cv2.imshow("Blured image", frame_blured)
        cv2.imshow("Edge detection", frame_processed)
        cv2.imshow("Binarized image", frame_binarized)

        try:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except ConnectionResetError:
            print('\nAi apasat q imbecilule\n')

    cam.release()  # comentam asta cand trecem la sockets
    cv2.destroyAllWindows()
    # # s.close()  se decomenteaza


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pinguu()
