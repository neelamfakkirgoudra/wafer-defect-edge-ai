import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------
# SETTINGS
# -------------------------
DATASET_DIR = "dataset/test"
NUM_CLASSES = 8
BATCH_SIZE = 16
MODEL_PATH = "best_model.pth"
DEVICE = torch.device("cpu")

# -------------------------
# TRANSFORMS (same as validation)
# -------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------
# LOAD TEST DATA
# -------------------------
test_data = datasets.ImageFolder(DATASET_DIR, test_transform)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False
)

class_names = test_data.classes

# -------------------------
# LOAD MODEL
# -------------------------
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------
# TEST EVALUATION
# -------------------------
all_preds = []
all_labels = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -------------------------
# METRICS
# -------------------------
accuracy = correct / total
precision = precision_score(all_labels, all_preds, average="macro")
recall = recall_score(all_labels, all_preds, average="macro")

print(f"Test Accuracy : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")

# -------------------------
# CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# -------------------------
# SAVE RESULTS
# -------------------------
model_size_mb = os.path.getsize("wafer_defect_model.onnx") / (1024 * 1024)

with open("test_metrics.txt", "w") as f:
    f.write(f"Test Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision     : {precision:.4f}\n")
    f.write(f"Recall        : {recall:.4f}\n")
    f.write(f"Model Size    : {model_size_mb:.2f} MB\n")
    f.write("Training Platform : CPU/GPU (as available)\n")
    f.write("Inference Platform: CPU\n")

print("✅ Test results saved successfully")
