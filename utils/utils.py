# app/utils.py

import torch
from PIL import Image
import torchvision.transforms as transforms
import io
from facenet_pytorch import MTCNN
import os
import uuid  # For generating unique filenames

# Initialize MTCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(
    image_size=224,
    margin=0,
    min_face_size=20,
    device=device
)

# Define the directory to save pre-processed images
PREPROCESSED_DIR = 'pre-processed_images'

# Ensure the directory exists
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

def get_transforms():
    """
    Returns the transformation pipeline applied to input images.
    
    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean for RGB
            std=[0.229, 0.224, 0.225]    # ImageNet std for RGB
        )
    ])

def preprocess_image(image_bytes):
    """
    Preprocesses the input image bytes by detecting and cropping the face.
    Converts grayscale images to RGB by duplicating channels.
    Saves the cropped face and returns the transformed tensor.
    
    Args:
        image_bytes (bytes): Image file in bytes.
    
    Returns:
        torch.Tensor: Preprocessed face image tensor ready for the model.
    """
    # Open the image as RGB
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Detect faces and probabilities using MTCNN
    boxes, probs = mtcnn.detect(image)
    
    if boxes is None or len(boxes) == 0:
        raise ValueError("No face detected in the image.")
    
    # Assuming the first detected face
    box = boxes[0]
    prob = probs[0]
    
    if prob < 0.90:
        raise ValueError("Low confidence in face detection.")
    
    # Crop the face using the detected box
    face = image.crop(box)
    
    # Convert to grayscale
    face = face.convert('L')
    
    # Convert back to RGB by duplicating channels
    face = face.convert('RGB')
    
    # Save the cropped face with confidence
    save_cropped_face(face, prob)
    
    # Apply transformations
    transform = get_transforms()
    face_tensor = transform(face)
    
    # Add batch dimension
    face_tensor = face_tensor.unsqueeze(0)
    
    return face_tensor

def save_cropped_face(face_image, confidence):
    """
    Saves a cropped face image to the pre-processed_images directory with a unique filename.
    
    Args:
        face_image (PIL.Image.Image): Cropped face image.
        confidence (float): Confidence score of the face detection.
    """
    # Generate a unique filename using UUID
    unique_id = uuid.uuid4().hex
    filename = f"face_{unique_id}.jpg"
    filepath = os.path.join(PREPROCESSED_DIR, filename)
    
    # Save the image
    face_image.save(filepath, format='JPEG')
    
    # Log the saving action
    print(f"Cropped face saved to {filepath} with MTCNN confidence {confidence:.2f}")
