
import torch
import torchvision.transforms as transforms
from PIL import Image
import warnings
import cv2
import time
import os
import psutil
from timmML2.models.factory import create_model

warnings.filterwarnings("ignore")

def preprocess_image(image):
    # Convert the OpenCV image (BGR) to PIL format (RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    c_size = 256
    if img.size[0] < c_size * 2:
        img = img.resize((c_size * 2, int(c_size * 2 * img.size[1] / img.size[0])))
    if img.size[1] < c_size * 2:
        img = img.resize((int(c_size * 2 * img.size[0] / img.size[1]), c_size * 2))

    kk = 14
    if img.size[0] > c_size * kk:
        img = img.resize((c_size * kk, int(c_size * kk * img.size[1] / img.size[0])))
    if img.size[1] > c_size * kk:
        img = img.resize((int(c_size * kk * img.size[0] / img.size[1]), c_size * kk))
    return img

# Initialize the Raspberry Pi camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Load the model
model_load_start_time = time.time()
model = create_model('efficientnet_lite2')
PATH_model = "teacher.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(PATH_model, map_location=device))
model.eval()
model_load_end_time = time.time()

# Metrics calculations
#ram_before = psutil.virtual_memory().used / (1024 ** 2)  # RAM in MB
ram_before = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # RAM in MB
disk_before = psutil.disk_usage('/').used / (1024 ** 3)  # Disk space used in GB
cpu_percent_before = psutil.cpu_percent(interval=1)

# Define image transformations
MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_STD[0], MEAN_STD[1])
])
for i in range(100):
#while True:
    # Capture an image
    ret, image = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture image")
    
    img = preprocess_image(image)

    # Perform image transformations
    input_tensor = img_transform(img)
    input_batch = input_tensor.unsqueeze(0)

    # Generate and print output
    inference_start_time = time.time()
    with torch.no_grad():
        output, features = model(input_batch)
    pred_count = round(torch.sum(output).item())
    inference_end_time = time.time()
    #print(f"Predicted count: {pred_count}")
    #print(f"Inference time: {inference_end_time - inference_start_time:.2f} seconds")
    
    # Sleep for 30 seconds before capturing the next image
    # time.sleep(30)

ram_after = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # RAM in MB
#ram_after = psutil.virtual_memory().used / (1024 ** 2)  # RAM in MB
disk_after = psutil.disk_usage('/').used / (1024 ** 3)  # Disk space used in GB
cpu_percent_after = psutil.cpu_percent(interval=1)

ram_usage = ram_after - ram_before
disk_usage = disk_after - disk_before
cpu_consumption = cpu_percent_after - cpu_percent_before
model_size = os.path.getsize(PATH_model) / (1024 ** 2)  # Model size in MB
model_load_time = model_load_end_time - model_load_start_time  # Time to load the model
throughput = 1 / (inference_end_time - inference_start_time)  # Inferences per second

print(f"RAM Usage: {ram_usage:.2f} MB")
print(f"Disk Usage: {disk_usage:.2f} GB")
print(f"CPU Consumption: {cpu_consumption:.2f}%")
print(f"Model Size: {model_size:.2f} MB")
print(f"Model Load Time: {model_load_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} inferences/second")
