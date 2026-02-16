import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# -------------------------
# SETTINGS
# -------------------------
DATASET_DIR = "hackathon_test_dataset"
MODEL_PATH = "best_model.pth"
NUM_CLASSES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)
print("Current Working Directory:", os.getcwd())

# -------------------------
# TRANSFORMS
# -------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------
# LOAD DATASET
# -------------------------
test_data = datasets.ImageFolder(DATASET_DIR, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

original_class_names = test_data.classes
print("Original Dataset Classes:", original_class_names)

# -------------------------
# LOAD TRAINED MODEL
# -------------------------
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------
# TRAINED CLASS ORDER
# -------------------------
trained_classes = ['Bridge', 'CMP', 'Clean', 'Cracks', 'LER', 'Opens', 'Others', 'Vias']

# Mapping dataset classes → trained classes
class_mapping = {}

for idx, class_name in enumerate(original_class_names):
    if class_name.lower() == "particle":
        class_mapping[idx] = trained_classes.index("Others")
    else:
        class_mapping[idx] = trained_classes.index(class_name)

# -------------------------
# INFERENCE
# -------------------------
all_preds = []
all_labels = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(len(labels)):
            true_label = class_mapping[labels[i].item()]
            pred_label = predicted[i].item()

            all_labels.append(true_label)
            all_preds.append(pred_label)

            if true_label == pred_label:
                correct += 1

            total += 1

# -------------------------
# METRICS
# -------------------------
accuracy = correct / total
precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

print("\nTotal Images  :", total)
print("Test Accuracy :", round(accuracy, 4))
print("Precision     :", round(precision, 4))
print("Recall        :", round(recall, 4))

# -------------------------
# CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=trained_classes,
            yticklabels=trained_classes,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Hackathon Test Dataset")
plt.tight_layout()

cm_path = os.path.join(os.getcwd(), "hackathon_confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print("Confusion matrix saved at:", cm_path)

# -------------------------
# SAVE LOG FILE
# -------------------------
log_path = os.path.join(os.getcwd(), "prediction_log.txt")

try:
    with open(log_path, "w") as f:
        f.write("====================================\n")
        f.write("Phase-2 Hackathon Prediction Log\n")
        f.write("====================================\n")
        f.write(f"Execution Date: {datetime.now()}\n")
        f.write(f"Dataset Used: {DATASET_DIR}\n")
        f.write(f"Total Images: {total}\n")
        f.write("------------------------------------\n")
        f.write(f"Accuracy  : {accuracy:.4f}\n")
        f.write(f"Precision : {precision:.4f}\n")
        f.write(f"Recall    : {recall:.4f}\n")
        f.write("------------------------------------\n")
        f.write(f"Inference Device: {DEVICE}\n")
        f.write(f"Model Used: {MODEL_PATH}\n")

    print("Log file saved at:", log_path)

except Exception as e:
    print("Error while saving log file:", e)

print("\n✅ Phase-2 prediction completed successfully!")
