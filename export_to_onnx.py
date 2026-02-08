import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

# -------------------------
# SETTINGS
# -------------------------
NUM_CLASSES = 8
MODEL_PATH = "best_model.pth"
ONNX_PATH = "wafer_defect_model.onnx"

# Always use CPU for ONNX export (safe & recommended)
DEVICE = torch.device("cpu")

# -------------------------
# LOAD MODEL ARCHITECTURE
# -------------------------
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Replace classifier (must match training)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------
# DUMMY INPUT
# -------------------------
# Shape: [batch, channels, height, width]
dummy_input = torch.randn(1, 3, 224, 224)

# -------------------------
# EXPORT TO ONNX
# -------------------------
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=18,          # IMPORTANT: avoids version conversion errors
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)

print("✅ ONNX model exported successfully!")
print(f"📁 Saved as: {ONNX_PATH}")
