from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import tensorflow as tf
import base64

app = Flask(__name__)
socketio = SocketIO(app)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='4.tflite')
interpreter.allocate_tensors()

EDGE_CONNECTIONS = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

def draw_keypoints_and_connections(frame, keypoints, confidence_threshold):
    frame_height, frame_width, _ = frame.shape
    keypoints_scaled = np.squeeze(np.multiply(keypoints, [frame_height, frame_width, 1]))  # Squeeze removes extra dimensions
    
    # Color map for edges
    color_map = {
        'm': (255, 0, 255), # magenta
        'c': (255, 255, 0), # cyan
        'y': (0, 255, 255)  # yellow
    }
    
    # Draw edges
    for edge, color_code in EDGE_CONNECTIONS.items():
        point1, point2 = edge
        y1, x1, conf1 = keypoints_scaled[point1]
        y2, x2, conf2 = keypoints_scaled[point2]
        if (conf1 > confidence_threshold) & (conf2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_map[color_code], 2)

    # Draw keypoints
    for keypoint in keypoints_scaled:
        keypoint_y, keypoint_x, keypoint_conf = keypoint
        if keypoint_conf > confidence_threshold:
            cv2.circle(frame, (int(keypoint_x), int(keypoint_y)), 4, (0, 255, 0), -1)  # Draw circle for keypoint

@socketio.on('frame')
def handle_frame(data):
    frame_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Resize and preprocess frame
    resized_frame = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
    input_image = tf.cast(resized_frame, dtype=tf.uint8)
    
    # Set input tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    
    # Get output tensor
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Draw keypoints and connections
    draw_keypoints_and_connections(frame, keypoints_with_scores, 0.4)

    _, buffer = cv2.imencode('.jpg', frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    emit('processed_frame', {'image': encoded_frame})

    print("Frame processed and emitted.")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Server started.")
    socketio.run(app, host='0.0.0.0', port=5000)
