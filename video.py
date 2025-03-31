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
    if not ret or frame is None:
        print("Error: Failed to capture frame!")
        continue

    # Ensure the frame has the correct number of channels before converting
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame  # If already grayscale, keep it as is

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Ensure frame size is valid before resizing
    if frame.shape[1] > 0 and frame.shape[0] > 0:
        scale_percent = 50  # Resize to 50% of original size
        width = max(1, int(frame.shape[1] * scale_percent / 100))
        height = max(1, int(frame.shape[0] * scale_percent / 100))
        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    else:
        print("Warning: Frame size is invalid, skipping resize.")
        resized_frame = frame  # Use original frame if resizing fails

    # Show result
    cv2.imshow("Live Face Detection", resized_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture and close windows
capture.release()
cv2.destroyAllWindows()