from flask import Flask, jsonify
from ultralytics import YOLO
import cv2

# 1️⃣ Create Flask app
app = Flask(__name__)

# 2️⃣ Load YOLO model
model = YOLO("best.pt")

# 3️⃣ Open webcam
cap = cv2.VideoCapture(0)

# 4️⃣ Define route
@app.route("/detect")
def detect():
    ret, frame = cap.read()
    if not ret:
        return jsonify({"predictions": []})

    results = model(frame, conf=0.3)

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x, y, w, h = box.xywh[0]
            detections.append({
                "class": label,
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h)
            })

    return jsonify({"predictions": detections})

# 5️⃣ Run the app
if __name__ == "__main__":
    app.run(debug=True)