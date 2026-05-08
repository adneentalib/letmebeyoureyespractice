from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)

# OIv7 model
model = YOLO("yolov8n-oiv7.pt")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "frame" not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400

    file = request.files["frame"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    results = model(frame, conf=0.05)

    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "label": model.names[class_id],
            "confidence": round(confidence, 3),
            "box": [x1, y1, x2, y2]
        })

    return jsonify({
        "detections": detections,
        "width": frame.shape[1],
        "height": frame.shape[0]
    })

if __name__ == "__main__":
    app.run(debug=True)