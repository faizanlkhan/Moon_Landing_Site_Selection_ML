# 🚀 Lunar Landing Site Selection with Reinforcement Learning  
**Adaptive Feature-Weight Optimization Using PPO**

This repository contains the code and workflow used in the research project on **learning adaptive feature weights** for lunar landing-site prediction. Instead of manually assigning weights to factors like water, slope, and sunlight, a **PPO-based reinforcement-learning agent** learns optimal weighting for each pixel on the lunar surface. The output includes a **suitability map**, **AI-derived weight maps**, and **high-quality landing zones** for mission planning.

---

## 📌 Key Idea  
This research replaces fixed weighting schemes with an RL policy that *learns* how important water, slope, and sunlight should be for safe and resource-rich landing-site evaluation.  
The model uses four aligned raster layers:

- 💧 **Water probability**  
- 🏔️ **Slope stability**  
- ☀️ **Sunlight fraction**  
- ⚠️ **Hazard intensity** (inverted so higher = safer)

The PPO agent outputs three continuous actions, converted via **softmax** into interpretable weights.

---

## 🧠 Method Overview  
### **RL Setup**
- **State:** `[water, slope, sunlight, hazard]`  
- **Action:** 3 continuous outputs → softmax → learned weights  
- **Suitability Score:**  
  ```
  S = w_water * f_water + w_slope * f_slope + w_sun * f_sunlight
  ```

### **Training**
- Algorithm: **PPO (Stable-Baselines3)**  
- Environment: Pixel-level sampling  
- Convergence typically around **1×10⁴ steps**

---

## 📈 Results  
- ⭐ **Mean suitability:** ~0.74 
- 📊 **Max suitability:** 0.9066  
- 🧭 **Average learned weights (top zones):**  
  - Water: ~0.595 
  - Slope: ~0.324  
  - Sunlight: ~0.080
- 🏁 Top landing zones extracted from the **95th percentile** of suitability.

---

## 🛠️ Installation  
```bash
pip install numpy scipy pandas rasterio rioxarray xarray geopandas shapely             matplotlib gymnasium stable-baselines3 torch scikit-image tqdm
```

---

## ▶️ How to Run  
1. **Add raster data** aligned to the illumination raster.  
2. **Run preprocessing** to normalize & stack features.  
3. **Train PPO**:  
   ```python
   model.learn(total_timesteps=20000)
   ```  
4. **Generate output maps**  
   - `suitability_map.tif`  
   - `ai_weights.tif`  
5. **Extract top landing zones** using 95th percentile thresholding.

---

## 📦 Outputs  
- `suitability_map.tif` — final suitability  
- `ai_weights.tif` — learned weight map  
- `top_zones.gpkg` — vector landing-zone polygons  
- PPO logs and training curves  

---

## 📚 Citation  
If you use or reference this work, please cite:

**“Adaptive Feature-Weight Optimization for Lunar Landing Site Selection Using Reinforcement Learning (2025).”**

---

