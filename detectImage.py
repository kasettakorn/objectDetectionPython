from imageai.Detection import ObjectDetection
import os
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "datasets/street.png"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
for eachObject in detections:
  print(eachObject["name"] , " : " , eachObject["percentage_probability"] )