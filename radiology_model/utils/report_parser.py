# radiology_model/utils/metrics.py
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, test_loader, device):
    y_true, y_pred = [], []
    model.eval()

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
