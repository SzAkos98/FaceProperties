import cv2
import numpy as np
import webcolors

blue_lower = np.array([100, 150, 0])
blue_upper = np.array([140, 255, 255])

brown_lower = np.array([10, 100, 20])
brown_upper = np.array([20, 255, 200])

green_lower = np.array([52, 0, 55])
green_upper = np.array([104, 255, 255])

face_cas = cv2.CascadeClassifier(r".\Assets\haarcascade_frontalface_default.xml")
eye_cas = cv2.CascadeClassifier(r".\Assets\haarcascade_eye.xml")
smile_cas = cv2.CascadeClassifier(r".\Assets\haarcascade_smile.xml")

def check_color(imghsv):
    blue_mask = cv2.inRange(imghsv, blue_lower, blue_upper)
    blue_px = np.sum(blue_mask != 0)
    brown_mask = cv2.inRange(imghsv, brown_lower, brown_upper)
    brown_px = np.sum(brown_mask != 0)
    green_mask = cv2.inRange(imghsv, green_lower, green_upper)
    green_px = np.sum(green_mask != 0)

    if green_px > brown_px and green_px > blue_px:
        return "Green"
    if brown_px > green_px and brown_px > blue_px:
        return "Brown"
    if blue_px > brown_px and blue_px > green_px:
        return "Blue"
    return "Unknown"

def find_color(requested_colour):  # finds the color name from RGB values

    min_colours = {}
    for name, key in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(name)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = key
        closest_name = min_colours[min(min_colours.keys())]
    return closest_name

def detectimg():
    img = cv2.imread(r".\Assets\test6.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces_rect = face_cas.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=9)

    for (x1, y1, w1, h1) in faces_rect:
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), thickness=2)

        smile_rect = smile_cas.detectMultiScale(
            gray[y1:y1 + w1, x1:x1 + h1], scaleFactor=1.8, minNeighbors=20)
        cv2.putText(img, "face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    2)

        for (x2, y2, w2, h2) in smile_rect:
            cv2.rectangle(img, (x1 + x2, y1 + y2), (x1 + x2 + w2, y1 + y2 + h2), (255, 0, 0), thickness=2)
            cv2.putText(img, "smile", (x1 + x2, y1 + y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                        2)

    eye_rect = eye_cas.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=9)

    for (x, y, w, h) in eye_rect:
        # Eye detect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        # Iris calc
        cx = x + w / 2
        cy = y + h / 2
        r = int((h + w) / 6)

        # Iris mask
        iris_mask = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(iris_mask, (int(cx), int(cy)), r, (255, 255, 255), -1)

        # Iris mask apply
        img3 = cv2.bitwise_and(img, img, mask=iris_mask)
        # cv2.imshow('Detected', img3)
        # cv2.waitKey(0)

        # Color saturation boost
        imghsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
        # imghsv[..., 1] = imghsv[..., 1] * 2

        color_id = check_color(imghsv)

        img2 = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)

        mean = cv2.mean(img2, iris_mask)
        print(mean)
        color = find_color(mean)
        cv2.putText(img, color_id + " " + color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12),
                    2)

    cv2.imshow('Detected faces', img)
    cv2.waitKey(0)
    return

def detectCam():
    cam = cv2.VideoCapture(0)
    while True:
        isNextFrameAvail, frame = cam.read()
        if not isNextFrameAvail:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces_rect = face_cas.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=9)

        for (x1, y1, w1, h1) in faces_rect:
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), thickness=2)

            smile_rect = smile_cas.detectMultiScale(
                gray[y1:y1+w1,x1:x1+h1], scaleFactor=1.8, minNeighbors=20)
            cv2.putText(frame, "face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

            for (x2, y2, w2, h2) in smile_rect:
                cv2.rectangle(frame, (x1+x2, y1+y2), (x1+x2 + w2, y1+y2 + h2), (255, 0, 0), thickness=2)
                cv2.putText(frame, "smile", (x1+x2, y1+y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                            2)

        eye_rect = eye_cas.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=9)

        for (x, y, w, h) in eye_rect:
            # Eye detect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

            # Iris calc
            cx = x + w / 2
            cy = y + h / 2
            r = int((h + w) / 6)

            # Iris mask
            iris_mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.circle(iris_mask, (int(cx), int(cy)), r, (255, 255, 255), -1)

            # Iris mask apply
            img3 = cv2.bitwise_and(frame, frame, mask=iris_mask)
            #cv2.imshow('Detected', img3)
            #cv2.waitKey(0)

            # Color saturation boost
            imghsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
            #imghsv[..., 1] = imghsv[..., 1] * 2

            color_id = check_color(imghsv)

            img2 = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)

            mean = cv2.mean(img2, iris_mask)
            print(mean)
            color = find_color(mean)
            cv2.putText(frame, color_id + " " + color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12),
                        2)

        cv2.imshow("PRESS Q TO EXIT", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return

def main():
    detectimg()
    detectCam()
    return

if __name__ == '__main__':
    main()
