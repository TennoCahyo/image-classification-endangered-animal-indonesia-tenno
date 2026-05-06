import io
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

IMG_SIZE = 150
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

CLASSES = ['Bekantan', 'Beruang Madu', 'Burung Rangkong', 'Gajah', 'Orangutan'] 

class ConvNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) 

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) 

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2) 

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 18 * 18, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def preprocess(image_bytes: bytes) -> torch.Tensor:
    """Mengubah byte gambar mentah menjadi Tensor yang siap dibaca model."""
   
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
   
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    tensor = transform(pil_img).unsqueeze(0) 
    return tensor


def postprocess(output_tensor: torch.Tensor) -> dict:

    probabilities = F.softmax(output_tensor, dim=1).squeeze()

    confidence, predicted_idx = torch.max(probabilities, 0)
    
    idx = predicted_idx.item()
    conf_score = confidence.item() * 100
    

    all_probs = {CLASSES[i]: round(probabilities[i].item() * 100, 2) for i in range(len(CLASSES))}
    
    if conf_score < 80.0:
        return {
            "class_index": -1, 
            "class_name": "Bukan Hewan Tersebut (Tidak Dikenali)",
            "confidence": round(conf_score, 2),
            "message": "Gambar kemungkinan besar bukan kelima hewan endemik tersebut.",
            "detail_probabilities": all_probs
        }
    else:
        return {
            "class_index": idx,
            "class_name": CLASSES[idx],
            "confidence": round(conf_score, 2),
            "message": "Berhasil mendeteksi hewan!",
            "detail_probabilities": all_probs
        }


def run_inference(model: torch.nn.Module, image_bytes: bytes, device: torch.device) -> dict:
 
    tensor = preprocess(image_bytes)
    tensor = tensor.to(device)

  
    start_time = time.time()
    model.eval() 
    with torch.no_grad(): 
        output = model(tensor)
    elapsed_ms = int((time.time() - start_time) * 1000)

    result = postprocess(output)
    result["inference_ms"] = elapsed_ms
    
    return result