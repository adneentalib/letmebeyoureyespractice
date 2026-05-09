from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import os
import gc

app = Flask(__name__)

MODEL_PATH = "yolov8n-oiv7.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "frame" not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400

    target_object = request.form.get("object", "mobile phone").strip().lower()

    file = request.files["frame"]
    image_bytes = file.read()

    if not image_bytes:
        return jsonify({"error": "Uploaded frame is empty"}), 400

    img_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    height, width = frame.shape[:2]

    detections = []
    feedback = f"{target_object} not found."
    distance_hint = "unknown"

    results = None
    result = None

    try:
        results = model.predict(
            frame,
            conf=0.05,
            imgsz=320,
            max_det=10,
            verbose=False
        )

        result = results[0]

        if result.boxes is not None:
            center_x = width / 2
            center_y = height / 2

            for box in result.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id].strip().lower()
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "label": label,
                    "confidence": round(confidence, 3),
                    "box": [
                        round(x1),
                        round(y1),
                        round(x2),
                        round(y2)
                    ]
                })

                if label == target_object:
                    object_center_x = (x1 + x2) / 2
                    object_center_y = (y1 + y2) / 2

                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    frame_area = width * height

                    if box_area > frame_area * 0.35:
                        distance_hint = "very close"
                    elif box_area > frame_area * 0.15:
                        distance_hint = "close"
                    elif box_area > frame_area * 0.05:
                        distance_hint = "medium"
                    else:
                        distance_hint = "far"

                    if box_area > frame_area * 0.35:
                        feedback = f"{target_object} found. Distance: {distance_hint}"
                    elif object_center_x < center_x - width * 0.15:
                        feedback = f"Move camera right. Distance: {distance_hint}"
                    elif object_center_x > center_x + width * 0.15:
                        feedback = f"Move camera left. Distance: {distance_hint}"
                    elif object_center_y < center_y - height * 0.15:
                        feedback = f"Tilt camera up. Distance: {distance_hint}"
                    elif object_center_y > center_y + height * 0.15:
                        feedback = f"Tilt camera down. Distance: {distance_hint}"
                    else:
                        feedback = f"Move forward. Distance: {distance_hint}"

                    break

    except Exception as e:
        return jsonify({"error": f"YOLO prediction failed: {str(e)}"}), 500

    finally:
        del img_array
        del frame

        if results is not None:
            del results

        if result is not None:
            del result

        gc.collect()

    return jsonify({
        "feedback": feedback,
        "target": target_object,
        "detections": detections,
        "width": width,
        "height": height,
        "distance_hint": distance_hint
    })


if __name__ == "__main__":
    app.run(debug=False)