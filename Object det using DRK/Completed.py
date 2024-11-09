import cv2
import numpy as np
import json

# Loading configuration
with open('config.json') as file:
    config = json.load(file)

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(config['yolo_cfg'], config['yolo_weights'])

# Load class names
with open(config['yolo_names'], 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# print(classes)
# Set video source
cap = cv2.VideoCapture(config['video_source'])  #  instead of this change this to 0 ---> webcam

while True:
    # Read a frame from the video source
    ret, frame = cap.read()
    # frame = cap.get(30)
    # frame = cap.get(30)
    if not ret:
        print("Failed to grab frame")
        break
    resized_frame = cv2.resize(frame, (480, 640))
    if cap.get(1) % 30 == 0: # ----> this part will be a part of drawing the box
        # resized_frame = cv2.resize(frame, (480, 640))  # You can change width and height
        height, width = resized_frame.shape[:2]
        # cv2.imshow("Vedo", resized_frame)
        # cv2.waitKey(20)
    #
        # Prepare the image for the model
        blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Get the output layer names
        layer_names = net.getLayerNames()

        # for i, name in enumerate(layer_names):
        #     print(f"Layer {i + 1}: {name}")

        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        # print(output_layers)

        # Run forward pass
        detections = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []
        person_detected = False
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # print("Scores: {}".format(scores))
                # print("class_id: {}".format(class_id))
                # print("confidence: {}".format(confidence))
                #
                if class_id == 0 and confidence > 0.7:  # Change confidence threshold if needed
                    person_detected = True
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, w, h) = box.astype("int")
                    x = int(center_x - (w / 2)) # this will be for top left
                    y = int(center_y - (h / 2))  # this will be for top right

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-Maxima Suppression, this will help to reduce
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        # Draw boxes on the resized frame
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                # label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                # if person_detected:
                label = f" person detected: {confidences[i]:.2f}"
                # else:
                #     label = f" person not detected: {confidences[i]:.2f}"

                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(resized_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if person_detected:
            print("person detected")
        else:
            print("person not detected")
    # Show the resized frame with detection
    cv2.imshow("Person Detected", resized_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
