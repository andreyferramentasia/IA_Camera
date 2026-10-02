import cv2
from ultralytics import YOLO

MODEL_PATH = 'yolov8n.pt'

# Índice da câmera: 0 = câmera padrão, 1 = segunda câmera (USB), 2, 3...
CAMERA_INDEX = 0

# Classes do COCO a detectar (pessoa + material escolar)
CLASSES = [
    0,   # person
    24,  # backpack (mochila)
    63,  # laptop
    64,  # mouse
    66,  # keyboard (teclado)
    67,  # cell phone (celular)
    73,  # book (livro/caderno)
    76,  # scissors (tesoura)
]

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir a câmera (índice {CAMERA_INDEX}). "
                       f"Tente mudar CAMERA_INDEX para 1 ou 2.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Câmera iniciada. Pressione 'Q' para sair.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame.")
            break

        results = model.predict(frame, conf=0.25, iou=0.45, classes=CLASSES, verbose=False)
        annotated = results[0].plot()

        n_boxes = len(results[0].boxes)
        cor = (0, 200, 0) if n_boxes > 0 else (0, 0, 255)
        cv2.putText(annotated, f"Objetos: {n_boxes}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

        annotated = cv2.resize(annotated, (1280, 720))
        cv2.imshow("Camera IA - YOLOv8", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Encerrado.")
