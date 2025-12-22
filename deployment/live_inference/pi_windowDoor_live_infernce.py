
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from tensorflow.keras.preprocessing import image
import time
import os
import psutil

# Initialize metrics
cpu_loads = []
ram_usages = []
start_time_overall = time.time()
frame_count = 0
model_path = 'Window_lightmodel_48.tflite'

model_size = os.path.getsize(model_path) / (1024 * 1024)  # in MB
print("Model Size:", model_size, "MB")

start_time = time.time()
interpreter = tflite.Interpreter(model_path=model_path)
end_time = time.time()
model_load_time = (end_time - start_time) * 1000  # Convert to milliseconds
print("Model Load Time:", model_load_time, "milliseconds")

total_inference_time = 0
inference_count = 0
interpreter.allocate_tensors()

inference_start_time = time.time()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
IMG_SIZE = 48

for i in range(100):
#while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    input_frame = input_frame.astype(np.float32)
    input_frame = (input_frame / 127.5) - 1
    input_frame = np.expand_dims(input_frame, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_frame)
    inference_end_time = time.time()
    total_inference_time += (inference_end_time - inference_start_time) * 1000  # Convert to milliseconds
    inference_count += 1

    interpreter.invoke()

    cpu_loads.append(psutil.cpu_percent())
    # ram_usages.append(psutil.virtual_memory().used / (1024 * 1024))
    ram_usages.append(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))  # Modified this line
    frame_count += 1

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = 'Open' if output_data >= 0 else 'Close'
    print('Predicted label:', predicted_label)
    # cv2.imshow('Live Inference', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
    #time.sleep(1)

cap.release()
cv2.destroyAllWindows()

if inference_count > 0:
    avg_inference = total_inference_time / inference_count
else:
    avg_inference = 0
print("Average Inference Time:", avg_inference, "milliseconds")

end_time_overall = time.time()
total_time = end_time_overall - start_time_overall

avg_cpu_load = sum(cpu_loads) / len(cpu_loads) if cpu_loads else 0
avg_ram_usage = sum(ram_usages) / len(ram_usages) if ram_usages else 0
throughput = frame_count / total_time

print("Average CPU Load:", avg_cpu_load, "%")
print("Average RAM Usage:", avg_ram_usage, "MB")
print("Throughput:", throughput, "frames per second")


