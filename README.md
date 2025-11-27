# This is the old first version, new version will be updated soon (Collab file)
# 🛰️ Moon Landing Site Prediction – Environment Setup Guide

This project uses Python and several scientific libraries for raster, image, and ML-based analysis to predict potential lunar landing sites.

Follow these steps carefully to set up your environment and install all required dependencies.

---

## 🧩 1. Prerequisites
Before you begin, make sure you have:
- Python 3.9+ installed → [Download Python](https://www.python.org/downloads/)
- pip (Python package manager)
- Git (optional, if cloning from GitHub)
- VS Code or any IDE of your choice

---

## ⚙️ 2. Create a Virtual Environment

### For Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
### For macOS/Linux:
```bash

python3 -m venv .venv
source .venv/bin/activate
```
You should now see (.venv) at the start of your terminal prompt.

### 📦 3. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 🧠 4. Install Required Dependencies
You can directly install all dependencies:

``` bash
pip install rasterio numpy opencv-python netCDF4 scikit-image scikit-learn matplotlib
```
Or, create a file named requirements.txt and paste this content:

rasterio
numpy
opencv-python
netCDF4
scikit-image
scikit-learn
matplotlib
Then install all at once:

```bash
pip install -r requirements.txt
```

### ✅ 5. Verify Installation

```bash
python -c "import rasterio, numpy, cv2, netCDF4, skimage, sklearn, matplotlib; print('✅ All dependencies installed successfully!')"
```

### 🚀 6. Run the Project
Make sure your terminal shows (.venv) before running the script:

```bash
python moon_landing_ml.py
```

