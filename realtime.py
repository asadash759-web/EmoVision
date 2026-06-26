import cv2
import numpy as np
from keras.models import load_model

# Load model
model = load_model('best_model.h5')
print("Model loaded successfully!")

# Emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)
print("Camera started! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Preprocess face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        # Predict emotion
        prediction = model.predict(face, verbose=0)
        emotion = emotions[np.argmax(prediction)]
        confidence = round(np.max(prediction) * 100, 2)

        # Display emotion on frame
        cv2.putText(frame, f'{emotion} ({confidence}%)',
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Face Emotion Recognition', frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
