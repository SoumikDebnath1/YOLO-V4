import numpy as np
import time
import cv2
import os
import imutils
import subprocess
from gtts import gTTS 
from pydub import AudioSegment

AudioSegment.converter = r"C:\Users\SOUMIK DEBNATH\Desktop\y4\yolov4-opencv-python\ffmpeg-6.0.tar.xz"

# Rest of your code...

# Specify the full path to the coco.names file
coco_names_path = "coco.names"

# Load the COCO class labels our YOLO model was trained on
LABELS = open(coco_names_path).read().strip().split("\n")

# Load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("C:\\Users\\SOUMIK DEBNATH\\Desktop\\y4\\yolov4-opencv-python\\yolov4.cfg", "C:\\Users\\SOUMIK DEBNATH\\Desktop\\y4\\yolov4-opencv-python\\yolov4.weights")

# Determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[layer - 1] for layer in net.getUnconnectedOutLayers()]


# Initialize
cap = cv2.VideoCapture(0)
frame_count = 0
start = time.time()
first = True
frames = []

while True:
    frame_count += 1
    # Capture frame-by-frameq
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frames.append(frame)

    if frame_count == 300:
        break
    if ret:
        key = cv2.waitKey(1)
        if frame_count % 60 == 0:
            end = time.time()
            # Grab the frame dimensions and convert it to a blob
            (H, W) = frame.shape[:2]
            # Construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            # Initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
            centers = []

            # Loop over each of the layer outputs
            for output in layerOutputs:
                # Loop over each of the detections
                for detection in output:
                    # Extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # Filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > 0.5:
                        # Scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # Use the center (x, y)-coordinates to derive the top and
                        # left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # Update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        centers.append((centerX, centerY))

            # Apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

            texts = []

            # Ensure at least one detection exists
            if len(idxs) > 0:
                # Loop over the indexes we are keeping
                for i in idxs.flatten():
                    # Find positions
                    centerX, centerY = centers[i][0], centers[i][1]
                    
                    if centerX <= W/3:
                        W_pos = "left "
                    elif centerX <= (W/3 * 2):
                        W_pos = "center "
                    else:
                        W_pos = "right "
                    
                    if centerY <= H/3:
                        H_pos = "top "
                    elif centerY <= (H/3 * 2):
                        H_pos = "mid "
                    else:
                        H_pos = "bottom "

                    texts.append(H_pos + W_pos + LABELS[classIDs[i]])

            print(texts)
            
            if texts:
                description = ', '.join(texts)
                tts = gTTS(description, lang='en')
                tts.save('tts.mp3')
                tts = AudioSegment.from_mp3("tts.mp3")
                subprocess.call(["ffplay", "-nodisp", "-autoexit", "tts.mp3"])

cap.release()
cv2.destroyAllWindows()
os.remove("tts.mp3")
