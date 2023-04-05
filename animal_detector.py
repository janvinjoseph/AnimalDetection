import cv2
import os

class AnimalDetector:

    def __init__(self):
        # Load Network
        net = cv2.dnn.readNet("dnn_model/yolov3.weights", "dnn_model/yolov3.cfg")
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(832, 832), scale=1 / 255)

        # Allow classes containing Animals only
        self.classes_allowed = [16, 17, 18, 19, 20, 21]

    def detect_animals(self, img):
        # Detect Objects
        animals_boxes = []
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
        for class_id, score, box in zip(class_ids, scores, boxes):
            if score < 0.5:
                # Skip detection with low confidence
                continue

            if class_id in self.classes_allowed:
                animals_boxes.append(box)

        # Print total count
        print("Total Animals Detected:", len(animals_boxes))

        return animals_boxes


if __name__ == '__main__':
    # Create Animal Detector Object
    detector = AnimalDetector()

    # Define image folder path
    images_folder_path = "images/"

    # Iterate through images in folder and detect animals
    for image_file in os.listdir(images_folder_path):
        # Read image
        image_path = os.path.join(images_folder_path, image_file)
        img = cv2.imread(image_path)

        # Detect animals
        animals_boxes = detector.detect_animals(img)

        # Draw bounding boxes on image
        for box in animals_boxes:
            cv2.rectangle(img, box, (0, 255, 0), 2)

        # Display image
        cv2.imshow("Image", img)
        cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()


