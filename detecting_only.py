from hamming.detect import detect_markers
from hamming.marker import *
import numpy as np
import cv2

if __name__ == '__main__':
    markers_detected = []
    obstacles_coordinates = []
    goal_coordinates = []
    boundaries_coordinates = []
    avaliable_markers = [2165, 733, 497, 4076, 1751, 1281, 1184]

    capture = cv2.VideoCapture(1)
    if capture.isOpened():  # try to get the first frame
        frame_captured, frame = capture.read()
    else:
        frame_captured = False

    while frame_captured:
        frame_captured, frame = capture.read()
        markers = detect_markers(frame)

        for marker in markers:
            marker.highlite_marker(frame)
            # if marker.id in avaliable_markers and marker.center not in obstacles_coordinates + goal_coordinates + boundaries_coordinates:
            #     if marker.id == 2165 or marker.id == 733:
            #         obstacles_coordinates.append(marker.center)
            #     elif marker.id == 497:
            #         goal_coordinates.append(marker.center)
            #     else:
            #         boundaries_coordinates.append(marker.center)

        print("Detected markers:", markers)
        print("Obstacles:", obstacles_coordinates)
        print("Goal:", goal_coordinates)
        print("Boundaries:", boundaries_coordinates)

        # Display the frame with highlighted markers
        cv2.imshow('AR capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('d'):
            break

    # Release the capture
    capture.release()
    cv2.destroyAllWindows()
