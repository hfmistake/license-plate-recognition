from ultralytics import YOLO
from predictions import Prediction
import easyocr

if __name__ == '__main__':
    IS_ENTRY_CAMERA = True
    VIDEO_FILE = "videos/1.mp4"

    vehicle_indexes = [2, 3, 5, 7]
    entry_line = [100, 150, 1800, 150]
    exit_line = [0, 370, 1800, 370]

    data = {
        "video_source": VIDEO_FILE,
        "line": entry_line if IS_ENTRY_CAMERA else exit_line,
        "object_indices": vehicle_indexes,
        "entry": IS_ENTRY_CAMERA,
        "pre_processing": True,
        "send_post": False,
        "preview": True
    }

    pre_trained = YOLO('models/weights/yolov8s.pt')
    plate_model = YOLO('models/treino5.pt')

    easyocr_reader = easyocr.Reader(['en'], gpu=True)

    plate_prediction = Prediction(plate_model, pre_trained, easyocr_reader, data)
    plate_prediction.predict()