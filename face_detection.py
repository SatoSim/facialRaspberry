import cv2
import subprocess
import time

# Step 1: Capture an image using raspistill
def capture_image():
    # Capture image using raspistill (saves as 'captured_image.jpg')
    subprocess.run(['raspistill', '-o', 'captured_image.jpg', '-t', '1', '-w', '640', '-h', '480'])
    time.sleep(1)  # Give some time to ensure the image is saved properly

# Step 2: Process the captured image
def process_image(image_path):
    # Load Haar cascade for face detection
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Check if cascade file loaded properly
    if faceCascade.empty():
        print("Error: Haar cascade file not loaded correctly!")
        return

    # Load the captured image
    image = cv2.imread(image_path)

    # Check if image loaded properly
    if image is None:
        print("Error: Image not found!")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Resize the image for better display
    scale_percent = 50  # Resize to 50% of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Step 3: Show the result
    cv2.imshow("Detected Faces", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()  # Capture an image from the camera
    process_image('captured_image.jpg')  # Process the image and detect faces
