import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict

class FER2013Model(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(FER2013Model, self).__init__()
        # Load a pre-trained EfficientNet model
        self.model = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
        # Replace the final fully connected layer with a new one (for FER2013)
        in_features = self.model._fc.in_features
        self.model._fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def load_model(model_path, device):
    """
    Loads the model's state_dict from the specified path and prepares the model for inference.
    
    Args:
        model_path (str): Path to the .pth file containing the state_dict.
        device (torch.device): Device to load the model onto.
    
    Returns:
        nn.Module: The loaded model ready for inference.
    """
    model = FER2013Model(num_classes=7, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel if necessary
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k.replace('module.', '')  # remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model