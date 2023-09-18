import cv2 as cv
import time
import pyttsx3
import threading

Conf_threshold = 0.4
NMS_threshold = 0.4
Min_score_for_voice = 0.7

COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv.dnn.readNet('yolov4.weights', 'yolov4.cfg')
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255, swapRB=True)

cap = cv.VideoCapture(0)
starting_time = time.time()
frame_counter = 0
detection_interval = 5  # Process every 5 frames

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

voice_feedback_given = {class_id: False for class_id in range(len(class_name))}
voice_lock = threading.Lock()

def run_voice_feedback(classid, score):
    if score >= Min_score_for_voice:
        with voice_lock:
            if not voice_feedback_given[classid]:
                labelname = class_name[classid]
                engine.say(labelname)
                engine.runAndWait()
                voice_feedback_given[classid] = True

def process_frame(frame):
    global frame_counter
    frame_counter += 1
    
    if frame_counter % detection_interval == 0:
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
        classes_detected = [cls for cls in classes]

        for (classid, score, box) in zip(classes, scores, boxes):
            class_index = int(classid)
            if class_index in classes_detected:
                color = COLORS[class_index % len(COLORS)]
                label = f"{class_name[class_index]} : {score:.2f}"

                if not voice_feedback_given[class_index]:
                    threading.Thread(target=run_voice_feedback, args=(class_index, score)).start()

                cv.rectangle(frame, box, color, 1)
                cv.putText(frame, label, (box[0], box[1] - 10),
                           cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)

        for class_id in range(len(class_name)):
            if class_id not in classes_detected:
                voice_feedback_given[class_id] = False

    cv.imshow('frame', frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    process_frame(frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
