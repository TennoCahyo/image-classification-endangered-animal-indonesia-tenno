import sys
import os
import torch
import torch.nn as nn
from torch.optim import Adam


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app.model import ConvNet
from src.preprocessing.data_loader import get_data_loaders
from src.evaluation.evaluate import evaluate_model

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device yang digunakan: {device}")

    
    train_path = '/content/drive/MyDrive/TugasDSAI/Cleaned Endangered Animal/Data_Train'
    test_path = '/content/drive/MyDrive/TugasDSAI/Cleaned Endangered Animal/Data_Tes'
    
    #Load Data
    print("Sedang memuat data...")
    train_loader, test_loader = get_data_loaders(train_path, test_path)
    
    train_count = len(train_loader.dataset)
    test_count = len(test_loader.dataset)
    print(f'Train Count: {train_count}')
    print(f'Test Count: {test_count}')

   
    model = ConvNet(num_classes=5).to(device)
    
   
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()
    num_epochs = 10
    
    best_f1_score = 0.0

    # Training Loop
    print("Memulai proses training...")
    for epoch in range(num_epochs):
        
       
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_accuracy += int(torch.sum(prediction == labels.data))

        # Hitung rata-rata akurasi dan loss training
        train_accuracy = train_accuracy / train_count
        train_loss = train_loss / train_count

      
        test_accuracy, f1 = evaluate_model(model, test_loader, device)

        print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f} | Test F1 Score: {f1:.4f}')

   
        if f1 > best_f1_score:

            if not os.path.exists('models'):
                os.makedirs('models')
            
            # Simpan file .pth
            torch.save(model.state_dict(), 'models/best_checkpoint.pth')
            best_f1_score = f1
            print(f"--> Model terbaik disimpan dengan F1 Score: {f1:.4f}")


if __name__ == '__main__':
    main()