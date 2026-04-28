
from hamming.detect import detect_markers
from hamming.marker import *
import numpy as np

if __name__ == '__main__':

    markers_detected = []
    boundaries_coordinates = []
    items_list = []
    obstacles_coordinates = []
    goal_coordianets = []
    detected =[]
    avaliable_markers =[2165,733,497,4076,1751,1281,1184]

    capture = cv2.VideoCapture(0)
    if capture.isOpened():  # try to get the first frame
        frame_captured, frame = capture.read()
    else:
        frame_captured = False
    while frame_captured:
        frame_captured, frame = capture.read()

        if len(markers_detected) != 1:
            markers = detect_markers(frame)
            print(detected)
            for marker in markers:
                marker.highlite_marker(frame)
                if (len(markers) == 7 ): #numer ob ar_markers and len(detected)==7
                    markers_detected.append(markers)
                    break # after you find 7 markers, stop detecting

        print("detected markers:", markers_detected)

        obstacles = [2165,733]
        goal = [497]
        boundaries = [4076,1751,1281,1184]


        for myList in markers_detected:
            for item in myList:
                if item.id in obstacles and item.center not in obstacles_coordinates and item.id not in detected:
                    obstacles_coordinates.append(item.center)
                    detected.append(item.id)
                if item.id in goal and item.center not in goal_coordianets and item.id not in detected:
                    goal_coordianets.append(item.center)
                    detected.append(item.id)
                if item.id in boundaries and item.center not in boundaries_coordinates and item.id not in detected:
                    boundaries_coordinates.append(item.center)
                    detected.append(item.id)
                    first = boundaries_coordinates[0]
                    last = boundaries_coordinates[-1]
                if item not in items_list:
                    items_list.append(item)


        print("obstacles:", obstacles_coordinates)
        print("goal:", goal_coordianets)
        print("coordination list: ", boundaries_coordinates)
        print("items list:", items_list)

        boundaries_coordinates1 = np.array(boundaries_coordinates)
        for point1, point2 in zip(boundaries_coordinates1, boundaries_coordinates1[0:]):
            cv2.line(frame, point1, point2, [255, 0, 150], 2)
            cv2.line(frame, first, last, [255, 0, 150], 2)

        with open('txt_files/boundaries_detector.txt', 'w') as file:
            for x, y in boundaries_coordinates:
                file.write('{} {}\n'.format(x, y))
        with open('txt_files/goal_detector.txt', 'w') as file:
            for x, y in goal_coordianets:
                file.write('{} {}\n'.format(x, y))
        with open('txt_files/obstacles_detector.txt', 'w') as file:
            for x, y in obstacles_coordinates:
                file.write('{} {}\n'.format(x, y))

        # color_detection(frame) #for detecting green color
        cv2.imshow('AR capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('d'):
            break
        frame_captured, frame = capture.read()

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()


