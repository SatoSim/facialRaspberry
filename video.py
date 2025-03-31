import cv2

# Load Haar cascade for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if cascade file loaded properly
if faceCascade.empty():
    print("Error: Haar cascade file not loaded correctly!")
    exit()

# Open video capture (0 for default camera)
capture = cv2.VideoCapture(0)

# Check if camera opened successfully
if not capture.isOpened():
    print("Error: Could not open camera!")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = capture.read()
    if not ret:
        print("Error: Failed to capture frame!")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Resize frame for display
    scale_percent = 50  # Resize to 50% of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # Show result
    cv2.imshow("Live Face Detection", resized_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture and close windows
capture.release()
cv2.destroyAllWindows()
