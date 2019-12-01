import numpy as np
import argparse
import dlib
import cv2
from imutils.video import VideoStream
import time
from scipy.spatial import distance
from threading import Thread
from playsound import playsound

def rect_to_bb(rect):
    """Function to convert rectangle bounding box into a tuple"""
    x=rect.left()
    y=rect.top()
    width=rect.right()-x
    height=rect.bottom()-y

    return (x, y, width, height)


def convert_shape_to_array(shape, dtype="int"):
    """Function to convert shape variable to numpy array"""
    coords = np.zeros(shape=(68,2),dtype=dtype)

    for i in range(0,68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def resize_image(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    """Function to resize the image"""
    dim  = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is not None:
        r = width / float(w)
        dim = (width, int(h*r))
    else:
        r = height/ float(h)
        dim = (int(w*r), height)

    resized = cv2.resize(image, dim, interpolation = interpolation)

    return resized


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear= (A+B)/(2*C)

    return ear

def detect_landmarks(args):
    """Function to detect facial landmarks"""
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args['shape_predictor'])

    vs = VideoStream().start()
    time.sleep(0.2)
    while True:
        frame = vs.read()
        #image = cv2.imread(args['image'])
        image = resize_image(frame, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        bb = face_detector(gray, 0)

        for (i, rect) in enumerate(bb):
            shape = predictor(gray, rect)
            shape = convert_shape_to_array(shape)
            (x, y, w, h) = rect_to_bb(rect)
            cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0), 2)

            # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            for (x,y) in shape:
                cv2.circle(image, (x,y),1,(0,0,255),-1)

        cv2.imshow("Frame",image)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


def detect_blinks(args):
    """Functixon to detect blinks with EAR"""

    eye_ar_theshold = 0.3
    eye_ar_frames = 3

    counter = 0
    blinks = 0

    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args['shape_predictor'])

    vs = VideoStream().start()
    time.sleep(0.2)
    while True:
        frame = vs.read()
        image = resize_image(frame, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        bb = face_detector(gray, 0)

        for (i, rect) in enumerate(bb):
            shape = predictor(gray, rect)
            shape = convert_shape_to_array(shape)

            # Calculate Eye aspect ratio for both the eyes
            leftEye = shape[36:42]
            rightEye = shape[42:48]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # Average EAR for both eyes
            ear = (leftEAR+rightEAR)/2

            # Counter for blinks
            if ear < eye_ar_theshold:
                counter += 1
            else:
                if counter >= eye_ar_frames:
                    blinks += 1

                counter = 0

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(image, "Blinks: {}".format(blinks), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame",image)
        key = cv2.waitKey(1) & 0xFF


        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


def play_sound(path):
    """Function to play an alarm sound"""
    playsound(path)


def drowsiness_detection(args):
    """Function to detect drowsiness and raise an alert"""

    eye_ar_theshold = 0.3
    eye_ar_frames = 48

    counter = 0
    alarm_on = False

    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args['shape_predictor'])

    vs = VideoStream().start()
    time.sleep(0.2)
    while True:
        frame = vs.read()
        image = resize_image(frame, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        bb = face_detector(gray, 0)

        for (i, rect) in enumerate(bb):
            shape = predictor(gray, rect)
            shape = convert_shape_to_array(shape)

            # Calculate Eye aspect ratio for both the eyes
            leftEye = shape[36:42]
            rightEye = shape[42:48]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # Average EAR for both eyes
            ear = (leftEAR+rightEAR)/2

            if ear < eye_ar_theshold:
                counter += 1

                if counter >= eye_ar_frames:
                    if not alarm_on:
                        alarm_on = True

                        if args['sound'] != "":
                            t = Thread(target=play_sound, args=(args['sound'],))
                            t.daemon = True
                            t.start()
                    cv2.putText(image, "DROWSINESS ALERT", (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                counter = 0
                alarm_on = False

            cv2.putText(image, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
    parser.add_argument("-i", "--image", required=False, help="path to input image")
    parser.add_argument("-s", "--sound", required=True, help="path to input image")
    args = vars(parser.parse_args())
    drowsiness_detection(args)















