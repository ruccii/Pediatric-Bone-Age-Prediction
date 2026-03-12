
import cv2
import numpy as np
from torchvision import transforms
from models import channel_mean, channel_std

# Preprocessing function for uploaded image
def preprocess_image(image, resize_dim=(512, 512)):
    # Convert PIL image to NumPy array
    img = np.array(image)
    
    # Check the number of channels
    if len(img.shape) == 3:  # RGB or RGBA image
        if img.shape[2] in [3, 4]:  # RGB (3 channels) or RGBA (4 channels)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unexpected number of channels in input image: {img.shape[2]}")
    elif len(img.shape) == 2:  # Already grayscale
        pass  # No conversion needed
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    
    # Bilateral filtering
    img_filtered = cv2.bilateralFilter(img_clahe, 9, 75, 75)
    
    # Histogram equalization
    img_equalized = cv2.equalizeHist(img_filtered)
    
    # Resize without padding
    img_resized = cv2.resize(img_equalized, resize_dim, interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize
    img_normalized = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_normalized, img_resized  # Return both for display

# Transform for model input
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[channel_mean], std=[channel_std])
])