import cv2
import numpy as np

# Load the pre-trained MobileNet model
net = cv2.dnn.readNetFromCaffe('mobilenet_v2_deploy.prototxt', 'mobilenet_v2.caffemodel')

# Define a simple function to map ImageNet class IDs to 'cat' or 'dog'
def get_class_label(class_id):
    if 151 <= class_id <= 268:
        return 'dog'
    elif 281 <= class_id <= 285:
        return 'cat'
def classify_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if image is loaded successfully
    if image is None:
        print(f"Error: Could not load image from path: {image_path}")
        return

    # Prepare the image for classification by resizing and creating a blob
    blob = cv2.dnn.blobFromImage(image, 0.017, (224, 224), (104, 117, 123))
    net.setInput(blob)

    # Perform classification
    predictions = net.forward()

    # Get the class with the highest confidence score
    class_id = np.argmax(predictions[0])
    confidence = float(predictions[0][class_id])
  
     #Get the class label
    label = get_class_label(class_id)

    # Print the values for debugging
    print(f"Class ID: {class_id}, Confidence: {confidence}, Label: {label}")

    # Display the image with the classification result
    cv2.putText(image, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Classification', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Classify the image 1.jpeg
classify_image('2.jpg')