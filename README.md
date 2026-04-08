# Embryo-Dataset-Model-Benchmark
This project implements and compares deep learning models on embryo datasets using a custom loss function that combines sparse categorical cross-entropy with an additional domain-specific loss to improve prediction performance.
# 🧬 Human Embryo Phase Classification using Deep Learning

## 📌 Project Overview

This project implements a deep learning pipeline to classify **human embryo developmental phases** from time-lapse IVF images. The system compares **standard CrossEntropy loss** with a **custom distance-aware loss function** across multiple CNN architectures.

---

## 🎯 Objective

* Predict embryo developmental stages from image frames
* Compare **baseline vs custom loss functions**
* Evaluate performance across multiple CNN models
* Handle **class imbalance and ordinal nature of labels**

---

## 📂 Dataset Details

* Source: Kaggle Embryo Dataset
* Contains:

  * Time-lapse embryo images
  * CSV annotations for developmental phases

### 📁 Folder Structure Used

```
embryo_dataset/
   embryo_id/
      F0/
         image_1.png
         image_2.png
```

👉 Only **F0 focal plane** images are used for consistency.

---

## ⚙️ Data Preparation Pipeline

### 🔹 1. Dataset Loading

* Dataset downloaded using `kagglehub`
* Images accessed from:

  ```
  embryo_id → F0 → images
  ```

---

### 🔹 2. Data Sampling

* Frames sampled every **6 intervals**
* Dataset limited to **≤15,000 samples** for faster training

---

### 🔹 3. Data Splitting

Split is performed using **embryo IDs**:

* Train: ~70%
* Validation: ~15%
* Test: ~15%

👉 Prevents **data leakage**

---

### 🔹 4. Class Distribution Handling

Before merging:

* Class 15 has very few samples

After merging:

* Class 15 → merged into Class 14

👉 Final number of classes = **15**

---

### 🔹 5. Class Weights

Class imbalance is handled using:

$$
w_i = \frac{N}{C \cdot n_i}
$$

Where:

* $N$ = total samples
* $C$ = number of classes
* $n_i$ = samples in class $i$

---

## 🏷️ Classes

```
tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8,
t9+, tM, tSB, tB, tEB, tHB
```

---

## 🔄 Image Preprocessing

Transformations applied:

```
Resize(320×320) → CenterCrop(299×299)
```

### Why?

* Required for **InceptionV3 (299×299 input)**
* Maintains compatibility across all models

---

## 🧠 Models Used

| Model       | Description                    |
| ----------- | ------------------------------ |
| MobileNetV2 | Lightweight and efficient      |
| VGG16       | Deep CNN architecture          |
| VGG19       | Deeper VGG variant             |
| InceptionV3 | Multi-scale feature extraction |

---

## 🔥 Loss Functions

### 🔹 1. Cross Entropy Loss (Baseline)

$$
L = -\sum y \log(p)
$$

---

### 🔹 2. Custom Loss (Distance-Aware Loss)

$$
L = L_{CE} + \lambda \cdot \frac{|y_{true} - y_{pred}^{soft}|}{C}
$$

### 💡 Intuition:

* Treats classes as **ordered developmental stages**
* Penalizes predictions based on **distance between phases**
* Encourages biologically consistent predictions

---

## 🏋️ Training Setup

| Parameter     | Value |
| ------------- | ----- |
| Optimizer     | Adam  |
| Learning Rate | 3e-5  |
| Batch Size    | 32    |
| Epochs        | 4    |
| Device        | GPU   |

---
## 📈 Results
<img width="489" height="625" alt="image" src="https://github.com/user-attachments/assets/4e3d4cf4-d43d-4f9f-b2af-f6c21cbaeb54" />

---
## Conclusion 

This project implemented a deep learning pipeline for classification of human embryo developmental stages using multiple CNN architectures and compared Cross Entropy (CE) with a custom distance-aware loss function.

  Key Findings from Results:
The custom loss consistently outperformed Cross Entropy across all models, showing improvement in both accuracy and F1-score.
Best Performing Model: InceptionV3
CE → Acc = 0.2738, F1 = 0.2757
Custom → Acc = 0.2947, F1 = 0.2998
👉 Highest overall performance among all models
MobileNetV2 also performed competitively
Custom → Acc = 0.2922, F1 = 0.2965
👉 Good balance between efficiency and performance
VGG16 and VGG19 showed lower performance
Likely due to overfitting and higher model complexity
Still improved when using custom loss

---
## Key Insights


Incorporating ordinal relationships between embryo stages using the custom loss significantly improves model learning.
Lightweight and modern architectures (MobileNet, Inception) perform better than deeper traditional networks (VGG) on this dataset.
The dataset remains challenging due to:
High similarity between adjacent developmental stages
Limited training samples
     
---
