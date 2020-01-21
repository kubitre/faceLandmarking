import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2


# cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280, 720))

predictor_path = 'shape_predictor_81_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def getLandMark(urlImage):
    frame = cv2.imread(urlImage)
    resize_frame = cv2.resize(frame, (1920,1080))
    dets = detector(resize_frame, 0)
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        for num in range(shape.num_parts):
            cv2.circle(resize_frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (89,115,68), -1)
    cv2.imshow('frame', resize_frame)
    # out.write(frame)

# while(cap.isOpened()):
    # ret, frame = cap.read()
    # frame = cv2.flip(frame, 1)
getLandMark("_DSC1210.jpg")

# cap.release()
# out.release()

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed")
        # break
cv2.destroyAllWindows()
