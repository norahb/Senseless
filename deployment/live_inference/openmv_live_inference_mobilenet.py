import sensor, image, time, tf, uos, utime, gc

# Initialize the camera
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224)) # MobileNet typically takes 224x224 images
sensor.skip_frames(time=2000)

# Free memory before loading the model
free_memory_before = gc.mem_free()

# Record the start time for loading the model
start_load_time = utime.ticks_ms()

# Load the model from the SD card
model_path = 'Window_lightmodel_48.tflite' # Make sure the path is correct
net = tf.load(model_path, load_to_fb=True)

# Record the end time for loading the model
end_load_time = utime.ticks_ms()

# Calculate model load time (in milliseconds)
model_load_time_ms = utime.ticks_diff(end_load_time, start_load_time)

# Free memory after loading the model
free_memory_after = gc.mem_free()

# RAM usage (in bytes)
ram_usage = free_memory_before - free_memory_after
ram_usage_kb = ram_usage / 1024.0
print("RAM Usage (KB):", ram_usage_kb)
print("Model Load Time (ms):", model_load_time_ms) # Print model load time


def predict_image(img):
    # Run the image through the model
    predictions = net.classify(img)

    # Get the output tensor (assuming it's a list with a single value)
    prediction_output = predictions[0].output()[0]

    # Interpret the prediction
    predicted_class = 'Open' if prediction_output > 0 else 'Close'

    return predicted_class


# Lists to collect latency and throughput
latency_list = []
throughput_list = []

for i in range(1):
    # Capture an image
    img = sensor.snapshot()

    # Start time
    start_time = utime.ticks_ms()

    # Predict the class
    predicted_class = predict_image(img)

    # End time
    end_time = utime.ticks_ms()

    # Calculate elapsed time
    inference_time = utime.ticks_diff(end_time, start_time)

    # Calculate latency in seconds
    inference_time_seconds = inference_time / 1000.0

    # Calculate throughput in inferences per second
    throughput = 1 / inference_time_seconds

    # Append to lists
    latency_list.append(inference_time)
    throughput_list.append(throughput)

    print('Predicted Class:', predicted_class)

    # Wait for 1 second
    # time.sleep(1)

# Calculate and print average latency and throughput
average_latency = sum(latency_list) / len(latency_list)
average_throughput = sum(throughput_list) / len(throughput_list)

print('Average Latency (ms):', average_latency)
print('Average Throughput (inferences/s):', average_throughput)


