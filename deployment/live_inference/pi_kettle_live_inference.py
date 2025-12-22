import time
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import os
import psutil
import board
import busio
import adafruit_mlx90640
from scipy import ndimage

# Initialize MLX90640
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
mlx_shape = (24, 32)
mlx_interp_val = 10
mlx_interp_shape = (mlx_shape[0] * mlx_interp_val, mlx_shape[1] * mlx_interp_val)

# Performance metrics initialization
cpu_loads = []
ram_usages = []
start_time_overall = time.time()
frame_count = 0

model_size = os.path.getsize('Kettle_MobileNetV2_tflite_finetuned_48.tflite') / (1024 * 1024)  # in MB
print("Model Size:", model_size, "MB")

start_time = time.time()
interpreter = tflite.Interpreter(model_path='Kettle_MobileNetV2_tflite_finetuned_48.tflite')
end_time = time.time()
model_load_time = (end_time - start_time) * 1000  # Convert to milliseconds
print("Model Load Time:", model_load_time, "milliseconds")

total_inference_time = 0
inference_count = 0
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

frame = np.zeros(mlx_shape[0] * mlx_shape[1])

class_labels = ['Off', 'On']
predictions = []

#while True:
for i in range(100):
    try:
        mlx.getFrame(frame)
        data_array = np.fliplr(np.reshape(frame, mlx_shape))
        data_array = np.rot90(data_array, k=1)
        data_array = ndimage.zoom(data_array, mlx_interp_val)
        
        # Resize to 48x48 
        input_frame = cv2.resize(data_array, (48, 48))
        input_frame = (input_frame - np.min(input_frame)) / (np.max(input_frame) - np.min(input_frame))
        input_frame = (input_frame * 255).astype(np.uint8)
        input_frame = (input_frame / 127.5) - 1
        
        # Ensure the batch dimension exists and replicate the single channel three times
        input_frame = np.expand_dims(input_frame, axis=0)  
        input_frame = np.stack([input_frame, input_frame, input_frame], axis=-1).astype(np.float32)
        
        # Perform inference
        inference_start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_frame)
        interpreter.invoke()
        inference_end_time = time.time()
        total_inference_time += (inference_end_time - inference_start_time) * 1000  # Convert to milliseconds
        inference_count += 1

        # Get the output and print
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_label = 'On' if output_data >= 0.5 else 'Off'
        predictions.append(class_labels.index(predicted_label)) 
        
        print(f"Predicted Class: {predicted_label}")
        
        # Recording CPU and RAM metrics
        cpu_loads.append(psutil.cpu_percent())
        #ram_usages.append(psutil.virtual_memory().used / (1024 * 1024))
        ram_usages.append(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)) 
        frame_count += 1
        
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f'Exception: {e}')
        continue
        
    #time.sleep(1)

# Calculate performance metrics
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
