

import torch
from torch import nn
import torchvision.models as torchvision_models

# Model class definition
class ResNet50BoneAge(nn.Module):
    def __init__(self):
        super(ResNet50BoneAge, self).__init__()
        
        # Load pretrained ResNet50
        resnet = torchvision_models.resnet50(pretrained=False)  # Pretrained=False for loading state
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc = nn.Sequential(
            nn.Linear(2048 + 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        
    def forward(self, x, gender):
        features = self.features(x)
        features = torch.flatten(features, 1)
        combined = torch.cat([features, gender.unsqueeze(1)], dim=1)
        output = self.fc(combined)
        return output

# Load model function
def load_model():
    model_path = 'final_best_model.pth'
    try:
        # Explicitly set weights_only=False to avoid warning
        final_model_package = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure the file exists.")
    
    # Extract model state
    model_state = final_model_package.get('model_state_dict')
    if model_state is None:
        raise KeyError("Model file does not contain 'model_state_dict'.")
    
    # Extract normalization parameters from 'normalization_params' key
    normalization_params = final_model_package.get('normalization_params')
    if normalization_params is None:
        raise KeyError("Model file does not contain 'normalization_params'.")
    
    boneage_mean = normalization_params.get('boneage_mean')
    boneage_std = normalization_params.get('boneage_std')
    channel_mean = normalization_params.get('channel_mean')
    channel_std = normalization_params.get('channel_std')
    
    # Check if all parameters were found
    missing_params = [param for param, value in [
        ('boneage_mean', boneage_mean),
        ('boneage_std', boneage_std),
        ('channel_mean', channel_mean),
        ('channel_std', channel_std)
    ] if value is None]
    if missing_params:
        raise KeyError(f"Missing normalization parameters in 'normalization_params': {', '.join(missing_params)}")
    
    # Instantiate and load model
    model = ResNet50BoneAge()
    model.load_state_dict(model_state)
    model.eval()
    
    return model, boneage_mean, boneage_std, channel_mean, channel_std

# Load normalization params for use in other modules
model, boneage_mean, boneage_std, channel_mean, channel_std = load_model()