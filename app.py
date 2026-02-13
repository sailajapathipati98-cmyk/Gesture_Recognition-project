from flask import Flask, render_template_string, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle

# =========================
# Load trained KNN model
# =========================
with open("gesture_knn_model.pkl", "rb") as f:
    model = pickle.load(f)

GESTURE_NAMES = model.classes_

# =========================
# Mediapipe setup
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# Flask
# =========================
app = Flask(__name__)

# =========================
# Beautiful Frontend
# =========================
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>AI Gesture Recognition</title>
<style>
body {
    background: linear-gradient(135deg,#667eea,#764ba2);
    font-family: Arial;
    text-align:center;
    color:white;
}
h1 { margin-top:20px; }
img {
    border-radius:20px;
    box-shadow:0 0 30px rgba(0,0,0,0.4);
    margin-top:20px;
}
button {
    padding:12px 25px;
    margin:10px;
    font-size:18px;
    border:none;
    border-radius:8px;
    cursor:pointer;
}
.start { background:#00c853; color:white; }
.stop { background:#d50000; color:white; }
</style>
</head>

<body>
<h1>🤖 AI Gesture Recognition</h1>

<img id="video" width="720" src="">

<br>
<button class="start" onclick="start()">Start Camera</button>
<button class="stop" onclick="stop()">Stop Camera</button>

<script>
function start(){
 document.getElementById("video").src="/video";
}
function stop(){
 document.getElementById("video").src="";
}
</script>

</body>
</html>
"""

# =========================
# Camera Generator
# =========================
def generate():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        text = "No Hand"

        if result.multi_hand_landmarks:
            lm_list = []
            for lm in result.multi_hand_landmarks[0].landmark:
                lm_list.extend([lm.x, lm.y, lm.z])

            if len(lm_list) == 63:
                pred = model.predict([lm_list])[0]
                text = str(pred)

        cv2.putText(frame, text, (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')

# =========================
# Routes
# =========================
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video')
def video():
    return Response(generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(debug=True)
