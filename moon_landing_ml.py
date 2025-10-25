import rasterio
import numpy as np
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# ------------------------------
# Step 1: Load Elevation (.tif) and compute slope
# ------------------------------
with rasterio.open("data/elevation.tif") as src:
    elevation = src.read(1)

dy, dx = np.gradient(elevation)
slope = np.sqrt(dx**2 + dy**2)
flat_mask = slope < 5  # slope threshold for safe areas

# ------------------------------
# Step 2: Generate Crater Mask from YOLO Labels
# ------------------------------
def yolo_to_mask(image_shape, label_file):
    mask = np.zeros(image_shape, dtype=np.uint8)
    h, w = image_shape
    if not os.path.exists(label_file):
        return mask
    with open(label_file, 'r') as f:
        for line in f:
            cls, x_center, y_center, bw, bh = map(float, line.strip().split())
            x1 = int((x_center - bw/2) * w)
            y1 = int((y_center - bh/2) * h)
            x2 = int((x_center + bw/2) * w)
            y2 = int((y_center + bh/2) * h)
            mask[y1:y2, x1:x2] = 1
    return mask

# Combine all training labels into one mask
crater_mask = np.zeros(elevation.shape, dtype=np.uint8)
train_labels_path = "crater_data/LU3M6TGT_yolo_format/train/labels"
for file in os.listdir(train_labels_path):
    if file.endswith(".txt"):
        mask = yolo_to_mask(elevation.shape, os.path.join(train_labels_path, file))
        crater_mask = np.maximum(crater_mask, mask)

# Combined safe areas
combined_mask = flat_mask & (~crater_mask.astype(bool))  # True = safe

# ------------------------------
# Step 3: Load Sunlight (.jp2)
# ------------------------------
import rasterio
with rasterio.open('data/sunlight.jp2') as src:
    sunlight = src.read(1)

# Resize to match elevation shape
sunlight = resize(sunlight, elevation.shape)
sunlight = sunlight / sunlight.max()  # normalize 0-1

# ------------------------------
# Step 4: Prepare ML Features
# ------------------------------
X = np.stack([
    slope.flatten(),
    crater_mask.flatten(),
    sunlight.flatten()
], axis=1)

y = combined_mask.flatten().astype(int)

# ------------------------------
# Step 5: Train Random Forest
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Feature importances
importances = rf.feature_importances_
print("Feature importances:", importances)

# ------------------------------
# Step 6: Predict Full Map
# ------------------------------
y_full_pred = rf.predict(X)
pred_map = y_full_pred.reshape(elevation.shape)

# ------------------------------
# Step 7: Visualize and Save Output
# ------------------------------
os.makedirs("output", exist_ok=True)
plt.figure(figsize=(10,10))
plt.imshow(elevation, cmap="gray")
plt.imshow(pred_map, cmap="Greens", alpha=0.5)
plt.title("Predicted Safe Landing Zones")
plt.axis("off")
plt.savefig("output/safe_zone_map.png", bbox_inches='tight')
plt.show()