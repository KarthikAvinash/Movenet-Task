import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

interpreter = tf.lite.Interpreter(model_path='4.tflite')
interpreter.allocate_tensors()

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_keypoints_and_connections(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # Squeeze removes extra dimensions
    
    # Color map for edges
    color_map = {
        'm': (255, 0, 255), # magenta
        'c': (255, 255, 0), # cyan
        'y': (0, 255, 255)  # yellow
    }
    
    # Draw edges
    for edge, color in EDGES.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_map[color], 2)

    # Draw keypoints
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)  # Draw circle for keypoint

def update_threshold():
    global confidence_threshold
    confidence_threshold = float(threshold_entry.get())

def update_frame():
    ret, frame = cap.read()
    if ret:
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img,axis=0),192,192)
        input_image = tf.cast(img,dtype=tf.uint8)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'],np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        draw_keypoints_and_connections(frame,keypoints_with_scores,confidence_threshold)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        panel.img_tk = img_tk  # Keep reference to prevent garbage collection
        panel.config(image=img_tk)
    
    # Update frame every 5 milliseconds
    root.after(5, update_frame)  

def exit_app():
    root.destroy()
    cap.release()

cap = cv2.VideoCapture(0)




root = tk.Tk()
root.title("Task: Real-Time Pose Detection with Movenet Lightning")
root.geometry("800x700")

heading_label = tk.Label(root, text="Pose Detection with Movenet on Real Time Video Feed", font=("Arial", 18, "bold"))
heading_label.pack(pady=10)

threshold_frame = tk.Frame(root)
threshold_frame.pack()

threshold_label = tk.Label(threshold_frame, text="Threshold:", font=("Arial", 12))
threshold_label.grid(row=0, column=0)

threshold_entry = tk.Entry(threshold_frame, font=("Arial", 12), width=10)
threshold_entry.grid(row=0, column=1)

update_button = tk.Button(threshold_frame, text="Update Threshold", font=("Arial", 12), command=update_threshold)
update_button.grid(row=0, column=2, padx=10)

exit_button = tk.Button(root, text="Exit", font=("Arial", 12), command=exit_app)
exit_button.pack(pady=10)

confidence_threshold = 0.4  # Initial threshold
threshold_entry.insert(0, str(confidence_threshold))

panel = tk.Label(root)
panel.pack()

update_frame()  # Start updating the frame

creator_label = tk.Label(root, text="By: Karthik Avinash", font=("Arial", 10), fg="gray")
creator_label.place(relx=1, rely=1, anchor=tk.SE)

root.mainloop()
