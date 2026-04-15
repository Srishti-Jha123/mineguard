import cv2
from ultralytics import YOLO
import winsound

# Load your trained model
model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

# Improve camera quality
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Brightness improve
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)

    results = model(frame, conf=0.2)

    helmet = False
    vest = False
    gloves = False
    alert = False

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls].lower()

            color = (255, 255, 0)

            # ✅ PPE present
            if "helmet" in label and "no" not in label:
                helmet = True
                color = (0, 255, 0)

            elif "vest" in label and "no" not in label:
                vest = True
                color = (0, 255, 0)

            elif "glove" in label and "no" not in label:
                gloves = True
                color = (0, 255, 0)

            # ❌ PPE missing
            elif "no" in label:
                color = (0, 0, 255)
                alert = True

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 🔥 FINAL STATUS
    if helmet and vest and gloves:
        text = "Fully Protected"
        color = (0, 255, 0)
        alert = False
    else:
        text = "PPE Missing"
        color = (0, 0, 255)
        alert = True

    # 🔊 Continuous Beep if violation
    if alert:
        winsound.Beep(2500, 800)

    # Show status text
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("MineGuard PPE Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()