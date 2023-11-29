import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from models.unet.unet_class import Unet
device = "cpu"
model_pth = 'models/unet/Results/model_epoch4_fold_4.pth'

def load_model(model_path):

    model = Unet().to(device)
    model.load_state_dict(torch.load(model_path,map_location = torch.device(device)))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.33298134, 0.33009373, 0.29579783], [0.19390678, 0.18714967, 0.18701911])
    ])
    image = Image.open(image_path)
    image = transform(image)
    return image

def apply_threshold(prediction, threshold=0.1):
    prediction = prediction > threshold
    return prediction

def print_advancement(i):
    progress_e = 100 * i / 50
    print(f' Progress : {progress_e : .2f}%', end='\r')

# Load your trained model
model = load_model(model_pth)

# Directory containing your test images
test_images_root_dir = 'data/test_set_images'

# Directory to save the predictions
save_dir = 'models/unet/Results/preds'
os.makedirs(save_dir, exist_ok=True)

for i in range(1, 51):

    print_advancement(i)
    dir_path = os.path.join(test_images_root_dir, f'test_{i}')
    image_path = os.path.join(dir_path, f'test_{i}.png')  # Update if the naming is different

    image = preprocess_image(image_path)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        prediction = model(image)

    # Convert prediction to an image format
    # Update this part based on your prediction format
    prediction_image = prediction.squeeze().cpu().numpy()
    binary_prediction = apply_threshold(prediction_image)

    # Convert binary prediction to image and save
    binary_prediction_image = Image.fromarray((binary_prediction * 255).astype(np.uint8))
    save_path = os.path.join(save_dir, f'prediction_{i}.png')
    binary_prediction_image.save(save_path)