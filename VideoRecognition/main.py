

# pip install opencv-contrib-python # some people ask the difference between this and opencv-python
                                    # and opencv-python contains the main packages wheras the other
                                    # contains both main modules and contrib/extra modules
# pip install cvlib # for object detection

# # pip install gtts
# # pip install playsound
# use `pip3 install PyObjC` if you want playsound to run more efficiently.

import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
from food_facts import food_facts


def speech(text):
    print(text)
    language = "en"
    output = gTTS(text=text, lang=language, slow=False)
    output.save("C:\Informatica\KGP\Python\python_mega\DetectionObject\VideoRecognition\sounds\output.mp3")
    playsound("C:\Informatica\KGP\Python\python_mega\DetectionObject\VideoRecognition\sounds\output.mp3")


# Carica i file YOLO
net = cv2.dnn.readNet("C:\Informatica\KGP\Python\python_mega\DetectionObject\VideoRecognition\yoloConf\yolov4.weights", "C:\Informatica\KGP\Python\python_mega\DetectionObject\VideoRecognition\yoloConf\yolov4.cfg")

# Carica le classi COCO
with open("C:\Informatica\KGP\Python\python_mega\DetectionObject\VideoRecognition\yoloConf\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Definisci gli output layer di YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Avvia la videocamera
video = cv2.VideoCapture(0)
labels = []

while True:
    ret, frame = video.read()
    height, width, channels = frame.shape

    # Prepara l'immagine per YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Effettua la rilevazione
    outs = net.forward(output_layers)

    # Parsing dei risultati
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Coordinate del bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Coordinate dell'angolo superiore sinistro
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Applica NMS (Non-Maximum Suppression) per evitare box sovrapposti
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Disegna il bounding box e il label sull'immagine
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Aggiungi l'etichetta alla lista
            if label not in labels:
                labels.append(label)

    # Mostra il video con i bounding box
    cv2.imshow("Detection", frame)

    # Termina se premi 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Genera un report di rilevazione e fatti alimentari
i = 0
new_sentence = []
for label in labels:
    if i == 0:
        new_sentence.append(f"I found a {label}, and, ")
    else:
        new_sentence.append(f"a {label},")
    i += 1

speech(" ".join(new_sentence))
speech("Here are the food facts I found for these items:")

for label in labels:
    try:
        print(f"\n\t{label.title()}")
        food_facts(label)
    except:
        print(f"No food facts for {label}")

# Rilascia la videocamera e chiudi le finestre
video.release()
cv2.destroyAllWindows()




'''
import cv2
import numpy as np

# Percorso ai file di configurazione e pesi
cfg_file = 'C:\\Informatica\\KGP\\Python\\python_mega\\DetectionObject\\yolov4.cfg'
weights_file = 'C:\\Informatica\\KGP\\Python\\python_mega\\DetectionObject\\yolov4.weights'

# Inizializza la webcam o carica un video
cap = cv2.VideoCapture(0)  # Usa 0 per la webcam o inserisci il percorso di un video

# Carica il modello YOLO
net = cv2.dnn.readNet(weights_file, cfg_file)

# Imposta i nomi delle classi
with open("C:\\Informatica\\KGP\\Python\\python_mega\\DetectionObject\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crea un blob e passa al modello
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Ottieni i nomi degli output
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()

    # Gestisci la compatibilità tra le versioni
    if isinstance(output_layers_indices, np.ndarray):
        output_layers_indices = output_layers_indices.flatten()  # Appiattire se è un array 2D

    output_layers = [layer_names[i - 1] for i in output_layers_indices]

    # Esegui il rilevamento
    outs = net.forward(output_layers)

    # Analizza l'output
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]  # Indici delle classi
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Soglia di confidenza
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Coordinate del bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Applica Non-Maxima Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Disegna i bounding box
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # Mostra il frame
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''