import cv2
import threading

# Load Haar cascade for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if cascade file loaded properly
if faceCascade.empty():
    print("Error: Haar cascade file not loaded correctly!")
    exit()

# Open video capture (0 for default camera)
capture = cv2.VideoCapture(0)

# Reduce resolution for better performance
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not capture.isOpened():
    print("Error: Could not open camera!")
    exit()

frame = None

def capture_frames():
    global frame
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to capture frame!")
            break

# Start capturing in a separate thread
thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

while True:
    if frame is None:
        continue

    # Ensure the frame is in the correct format before converting
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame  # If it's already grayscale, keep it as is

    # Detect faces with optimized parameters
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show result
    cv2.imshow("Live Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
