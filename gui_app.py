import cv2
from tkinter import *
from PIL import Image, ImageTk
from ultralytics import YOLO
import winsound

# Load model
model = YOLO("best (1).pt")

cap = cv2.VideoCapture(0)
running = True

# Counters
helmet_count = 0
vest_count = 0
gloves_count = 0
violation_count = 0

# GUI
root = Tk()
root.title("MineGuard - Smart PPE System")
root.geometry("1200x700")
root.configure(bg="#121212")

# -------- LEFT PANEL --------
left_frame = Frame(root, bg="#1e1e2f", width=300)
left_frame.pack(side=LEFT, fill=Y)

title = Label(left_frame, text="MineGuard",
              font=("Arial", 24, "bold"),
              bg="#1e1e2f", fg="white")
title.pack(pady=20)

status_label = Label(left_frame,
                     text="SAFE",
                     font=("Arial", 20, "bold"),
                     bg="green", fg="white",
                     width=15)
status_label.pack(pady=20)

# Counters
helmet_label = Label(left_frame, text="Helmet: 0",
                     font=("Arial", 14),
                     bg="#1e1e2f", fg="white")
helmet_label.pack(pady=5)

vest_label = Label(left_frame, text="Vest: 0",
                   font=("Arial", 14),
                   bg="#1e1e2f", fg="white")
vest_label.pack(pady=5)

gloves_label = Label(left_frame, text="Gloves: 0",
                     font=("Arial", 14),
                     bg="#1e1e2f", fg="white")
gloves_label.pack(pady=5)

violation_label = Label(left_frame, text="Violations: 0",
                        font=("Arial", 14),
                        bg="#1e1e2f", fg="red")
violation_label.pack(pady=10)

# Buttons
def start():
    global running
    running = True

def stop():
    global running
    running = False

Button(left_frame, text="Start",
       bg="green", fg="white",
       font=("Arial", 12),
       width=15, command=start).pack(pady=10)

Button(left_frame, text="Stop",
       bg="red", fg="white",
       font=("Arial", 12),
       width=15, command=stop).pack()

# -------- RIGHT PANEL (VIDEO) --------
right_frame = Frame(root, bg="black")
right_frame.pack(side=RIGHT, expand=True, fill=BOTH)

video_label = Label(right_frame)
video_label.pack()

# -------- MAIN LOOP --------
def update_frame():
    global helmet_count, vest_count, gloves_count, violation_count

    if running:
        ret, frame = cap.read()

        if ret:
            results = model(frame, conf=0.2)

            alert = False

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    color = (0,255,0)

                    # Logic
                    if "no" in label.lower():
                        color = (0,0,255)
                        alert = True
                        violation_count += 1

                    elif "helmet" in label.lower():
                        helmet_count += 1

                    elif "vest" in label.lower():
                        vest_count += 1

                    elif "glove" in label.lower():
                        gloves_count += 1

                    # Draw
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, label, (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Alert
            if alert:
                winsound.Beep(1000, 200)
                status_label.config(text="VIOLATION", bg="red")
            else:
                status_label.config(text="SAFE", bg="green")

            # Update counters
            helmet_label.config(text=f"Helmet: {helmet_count}")
            vest_label.config(text=f"Vest: {vest_count}")
            gloves_label.config(text=f"Gloves: {gloves_count}")
            violation_label.config(text=f"Violations: {violation_count}")

            # Convert frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

    video_label.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
