import torch
from sklearn.metrics import f1_score

def evaluate_model(model, test_loader, device):
    model.eval()
    test_accuracy = 0.0
    all_predictions = []
    all_labels = []
    
    test_count = len(test_loader.dataset)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)

            test_accuracy += int(torch.sum(prediction == labels.data))

            all_predictions.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = test_accuracy / test_count
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    return test_accuracy, f1