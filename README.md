# 🏆 AI CUP 2025 Fall Competition  
### Computed Tomography Myocardium Image Segmentation II  
### Aortic Valve Object Detection  

**Team:** 8108  
- **Private Leaderboard:** 0.963030  
- **Public Leaderboard:** 0.971944  

---

## 📊 Dataset Overview
The dataset is divided into three subsets to ensure robust training and evaluation:

- **Training set:** Optimizes model parameters and learns discriminative features of the myocardium and aortic valve.  
- **Validation set:** Tunes hyperparameters and monitors generalization during training.  
- **Test set:** Reserved for final evaluation, ensuring unbiased performance assessment.  

| Set        | Patients       | Images  | Instances |
|------------|----------------|---------|-----------|
| Train      | Patient01–40   | 13,177  | 2,168     |
| Validation | Patient41–50   | 3,886   | 619       |
| Test       | Patient51–100  | 16,620  | –         |

---

## 🏗️ Model Architecture

### Backbone
- Input: **640×640×3** CT images  
- Convolutional layer followed by **MyGhost modules** for efficient channel expansion and reduced redundancy  
- **C2f blocks** integrated at each stage to improve gradient flow and feature representation  
- **WAFU modules** refine spatial information  
- Progressive resolution reduction: `160×160 → 80×80 → 40×40 → 20×20`  
- **SPPF module** aggregates multi-scale context for both small and large object detection  

### Neck
- Optimized **PANet** for multi-scale feature fusion  
- Deep features upsampled and concatenated with shallow features  
- Fusion stages enhanced with **C2f** and **WAFU modules**  
- Ensures balanced representation of local and global information  

### Head
- **Decoupled detection design**: separate branches for classification, bounding box regression, and objectness scoring  
- Attention modules applied at multiple scales:  
  - **SmallAttention:** 20×20×512  
  - **MediumAttention:** 40×40×256  
  - **LargeAttention:** 80×80×128  
- Multi-scale **Detect layer** integrates outputs from `[24, 37, 38, 39]` for robust predictions across object sizes  

---

## ⚡ Performance Metrics

| FPS (ms) | FLOPs (G) | Params (M) | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50:95 (%) |
|----------|-----------|------------|---------------|------------|------------|---------------|
| 0.2      | 60.0      | 16.5       | 91.1          | 94.1       | 95.6       | 68.4          |

---

## 🚀 Highlights
- High accuracy on both private and public leaderboards  
- Efficient backbone design with **Ghost + C2f + WAFU** modules  
- Robust multi-scale detection with attention mechanisms  
- Balanced trade-off between computational cost and detection performance  

---

📌 This README provides a structured overview of our solution for **AI CUP 2025 Fall Competition**.  
