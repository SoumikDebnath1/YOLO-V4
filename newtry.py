import cv2 as cv
import time
import pyttsx3

Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv.dnn.readNet('yolov3.weights', 'yolo_v3.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)


model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255, swapRB=True)

cap = cv.VideoCapture(0)
starting_time = time.time()
frame_counter = 0

engine = pyttsx3.init()
engine.setProperty('rate', 100)
engine.setProperty('volume', 1.0)

voice_feedback_given = {class_id: False for class_id in range(len(class_name))}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % 10 == 0:
        endingTime = time.time() - starting_time
        fps = frame_counter / endingTime
        cv.putText(frame, f'FPS: {fps:.2f}', (20, 50),
                   cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = f"{class_name[classid]} : {score:.2f}"

        if not voice_feedback_given[classid]:
            labelname = class_name[classid]
            engine.say(labelname)
            engine.runAndWait()
            voice_feedback_given[classid] = True

        cv.rectangle(frame, box, color, 1)
        cv.putText(frame, label, (box[0], box[1] - 10),
                   cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)

    for classid in voice_feedback_given.keys():
        if classid not in classes:
            voice_feedback_given[classid] = False

    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
