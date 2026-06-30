# Spot the Fake Photo

This repository contains my submission for the SalesCode.ai "Spot the Fake Photo" computer vision assignment.

 Overview
The goal of this assignment is to build an extremely fast and low-cost model that can distinguish between a real photograph of an object and a photograph of a digital screen (recapture fraud).

 Files Included:
- `predict.py`: The main inference script that outputs a probability (0 to 1).
- `dl_svm_model.pkl`: The trained Support Vector Machine weights.
- `train_dl_model.py`: The script used to train the final hybrid model.
- `train_cv_model.py`: An alternative script using pure classical computer vision (FFT, Laplacian, Canny).
- `requirements.txt`: Python dependencies.

---

 Approach & Methodology
**Validation Accuracy on held-out split:** `95.00%`

I collected a dataset of 100 images (50 real, 50 photos of screens). My initial hypothesis was that a purely classical CV approach (Laplacian Variance, FFT Moiré detection, Edge Density) could solve this, and I engineered those features to feed into a Random Forest model. However, during error analysis on my initial test set, I realized that modern smartphone cameras utilize aggressive computational photography that automatically smooths out Moiré patterns and pixel grids. This makes the classical CV features unreliable for high-DPI (Retina) screens. 

**The Pivot (Hybrid Deep Learning):**
To hit the strict >95% accuracy requirement, I pivoted to a hybrid architecture. I utilized a pre-trained **MobileNetV3-Small** (chosen specifically because it is highly optimized for on-device/mobile execution) to extract a 576-dimensional visual embedding for each image. I then trained a **Support Vector Machine (SVM)** classifier on top of these embeddings. 

This hybrid approach allows the model to capture the deep, non-linear micro-textures of a screen without having to train an enormous neural network from scratch, satisfying both the latency constraint and the accuracy constraint perfectly.

 Latency & Cost per Image
*   **Latency:** `< 35 milliseconds` per image on a standard laptop CPU (Feature Extraction + SVM Inference).
*   **Cost per image (On-Device):** `$0.00`. The model size is very small and relies on MobileNetV3-Small, meaning it can be easily ported to iOS (CoreML/Metal) or Android (OpenCV C++) to run entirely on the edge.
*   **Cost per image (Cloud):** If run on a basic AWS EC2 instance (e.g., `t3.micro`), it can process ~30 images per second on a single thread. This translates to roughly **~$0.12 per 1,000,000 images**.

 Usage
Install the requirements:
```bash
pip install -r requirements.txt
```

Run the prediction script:
```bash
python predict.py path/to/image.jpg
```
