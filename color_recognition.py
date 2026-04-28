import cv2
import numpy as np
center_point = [0,0]
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def color_detection(frame):
    global center_point
    kernel = np.ones((2, 2), np.uint8)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]

    ret, thresh = cv2.threshold(a_channel, 105, 255, cv2.THRESH_BINARY_INV)  # to binary
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)  # to get outer boundery only

    thresh = cv2.dilate(thresh, kernel, iterations=5)  # to strength week pixels
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        #cv2.drawContours(frame, contours, -1, (0, 255, 0), 5)
        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        x1 = x + w
        y1 = y + h
        # draw the biggest contour (c) in green
        cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
        center_point = [(int((x + x1) / 2), int((y + y1) / 2))]
        # coordinates of the rectangle
        print("Top left corner: ", x, y)
        print("Top right corner: ", x1, y)
        print("Bottom left corner: ", x, y1)
        print("Bottom right corner: ", x1, y1)
        print("Center point: ", center_point)



while True:
    _, frame = cap.read()
    # height, width, _ = frame.shape
    # mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv_frame, (36, 25, 25), (70, 255, 255))

    color_detection(frame)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('d'):
        with open('txt_files/start_detector.txt', 'w') as file:
            for x, y in center_point:
                file.write('{} {}\n'.format(x, y))

        break

cap.release()
cv2.destroyAllWindows()
