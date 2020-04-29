# USAGE
# import the necessary packages
import numpy as np
import argparse
import imutils
import sys
import cv2
import json

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained human activity recognition model")
# ap.add_argument("-c", "--classes", required=True,
# 	help="path to class labels file")
ap.add_argument("-i", "--video_name", type=str, default="",
    help="optional path to video file")
#args = vars(ap.parse_args(args=['--model','/Users/chenbingxu/PycharmProjects/c3d/Kinetics400_res3d/resnet-34_kinetics.onnx','--classes','/Users/chenbingxu/PycharmProjects/c3d/Kinetics400_res3d/action_recognition_kinetics.txt']))
#,'--input','/Users/chenbingxu/PycharmProjects/c3d/video_test/8 types of actions.mp4'
args = vars(ap.parse_args())
# load the contents of the class labels file, then define the sample
# duration (i.e., # of frames for classification) and sample size
# (i.e., the spatial dimensions of the frame)
CLASSES = open('/Users/chenbingxu/PycharmProjects/c3d/Kinetics400_res3d/action_recognition_kinetics.txt').read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

# load the human activity recognition model
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet('/Users/chenbingxu/PycharmProjects/c3d/Kinetics400_res3d/resnet-34_kinetics.onnx')

# grab a pointer to the input video stream
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["video_name"] if args["video_name"] else 0)

# loop until we explicitly break from it
submit = '/Users/chenbingxu/PycharmProjects/c3d/timeLabel.json'
result = []

while True:
    # initialize the batch of frames that will be passed through the
    # model
    frames = []

    # loop over the number of required sample frames
    for i in range(0, SAMPLE_DURATION):
        # read a frame from the video stream
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed then we've reached the end of
        # the video stream so exit the script
        if not grabbed:
            print("[INFO] no frame read from stream - exiting")
            sys.exit(0)

        # otherwise, the frame was read so resize it and add it to
        # our frames list
        frame = imutils.resize(frame, width=400)
        frames.append(frame)

    # now that our frames array is filled we can construct our blob
    blob = cv2.dnn.blobFromImages(frames, 1.0,
        (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
        swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    # pass the blob through the network to obtain our human activity
    # recognition predictions
    net.setInput(blob)
    outputs = net.forward()
    pro = np.max(outputs)
    label = CLASSES[np.argmax(outputs)]

    # loop over our frames
    for frame in frames:
        time = vs.get(cv2.CAP_PROP_POS_MSEC)
        # print(time)
        temp_dict = {}
        temp_dict["label"] = label
        temp_dict["time"] = time
        temp_dict["pro"] = 1-pro/100
        # result.append(temp_dict)
        # draw the predicted activity on the frame
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2)

        # display the frame to our screen
        cv2.imshow("Activity Recognition", frame)
        key = cv2.waitKey(1) & 0xFF



        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
		#
        # with open(submit, 'a') as f:
        #     json.dump(result, f)