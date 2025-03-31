import cv2

# Load Haar cascade for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if cascade file loaded properly
if faceCascade.empty():
    print("Error: Haar cascade file not loaded correctly!")
    exit()

# Load an image
image = cv2.imread("image.jpg")

# Check if image loaded properly
if image is None:
    print("Error: Image not found!")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = faceCascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Resize image for display
scale_percent = 50  # Resize to 50% of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Show result
cv2.imshow("Detected Faces", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
