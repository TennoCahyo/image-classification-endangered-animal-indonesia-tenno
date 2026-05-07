import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

def get_data_loaders(
 
    train_path='/content/drive/MyDrive/TugasDSAI/Cleaned Endangered Animal/Data_Train', 
    test_path='/content/drive/MyDrive/TugasDSAI/Cleaned Endangered Animal/Data_Tes',
    batch_size_train=64, 
    batch_size_test=32
):
    
    # Preprocessing & Augmentasi
    transformer = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])

    # Memuat Dataset Training
    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transformer),
        batch_size=batch_size_train, shuffle=True
    )
    
    # Memuat Dataset Testing
    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transformer),
        batch_size=batch_size_test, shuffle=True
    )
    
    return train_loader, test_loader