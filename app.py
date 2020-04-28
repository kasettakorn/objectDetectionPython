from imageai.Detection import VideoObjectDetection
import os
import tensorflow as tf
f_number = []
output = []
count = []
def forFrame(frame_number, output_array, output_count):
    if output_array:
        f_number.append(frame_number)
        output.append(output_array)
        count.append(output_count)❗️
        print("FOR FRAME" , frame_number)
        print("Output for each object : ", output_array)
        print("Output count for unique objects : ", output_count)
        print("------------END OF A FRAME --------------")
    return frame_number, output_array, output_count


execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "/models/yolo.h5"))
detector.loadModel()

detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "datasets/city.mp4"),
    output_file_path=os.path.join(execution_path, "city_result"), 
    frames_per_second=29, 
    log_progress=True,
    per_frame_function=forFrame,
    minimum_percentage_probability=60)