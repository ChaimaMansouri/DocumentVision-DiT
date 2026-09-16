# 📄 DocumentVision-DiT

### Document Image Classification with DiT and Transfer Learning

A deep learning project for **automatic document image classification** using a pretrained **Document Image Transformer (DiT)** model fine-tuned on the **Tobacco3482** dataset.

The system classifies document images into **10 predefined categories** using transfer learning and fine-tuning.

---

## 🎯 Project Overview

The project aims to automatically recognize document types from images, reducing the need for manual document organization.

### Main Pipeline

```text
Tobacco3482 Dataset
        ↓
Image Collection
        ↓
Stratified Train / Validation Split
        ↓
AutoImageProcessor
        ↓
Pretrained DiT
        ↓
Fine-Tuning
        ↓
10-Class Classification
        ↓
Validation Evaluation
```

---

## 🧠 Model

The project uses:

**Microsoft DiT — Document Image Transformer**

Pretrained model:

```text
microsoft/dit-base-finetuned-rvlcdip
```

The pretrained model is adapted to the **10 Tobacco3482 classes**.

The original classifier has **16 classes**, so the classification layer is reinitialized for the 10 target classes.

---

## 📊 Dataset

The project uses the **Tobacco3482** dataset.

* **3,482 document images**
* **10 document categories**

### Classes

| ID | Category   |
| -: | ---------- |
|  0 | ADVE       |
|  1 | Email      |
|  2 | Form       |
|  3 | Letter     |
|  4 | Memo       |
|  5 | News       |
|  6 | Note       |
|  7 | Report     |
|  8 | Resume     |
|  9 | Scientific |

---

## 🔄 Data Preparation

The notebook:

1. Loads the dataset.
2. Collects image paths and labels.
3. Encodes categories into numerical IDs.
4. Performs an **80/20 stratified train-validation split**.
5. Processes images using the pretrained model's `AutoImageProcessor`.

```python
train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)
```

### Image Processing

No manual augmentation pipeline is implemented in the notebook. Images are prepared through the DiT-associated `AutoImageProcessor`.

---

## ⚙️ Training Configuration

| Parameter          | Value                                  |
| ------------------ | -------------------------------------- |
| Model              | `microsoft/dit-base-finetuned-rvlcdip` |
| Dataset            | Tobacco3482                            |
| Classes            | 10                                     |
| Train / Validation | 80 / 20                                |
| Batch size         | 8                                      |
| Learning rate      | `2e-5`                                 |
| Epochs             | 10                                     |
| Weight decay       | `0.01`                                 |
| Random state       | 42                                     |
| Mixed precision    | FP16 when CUDA is available            |
| Framework          | PyTorch / Hugging Face Transformers    |

---

## 📈 Results

The model was trained for **10 epochs** and evaluated on the validation set.

### Final Validation Performance

* **Validation Accuracy:** **96.27%**
* **Evaluation Loss:** **0.3530**

The highest observed validation accuracy during training was:

**96.56% at Epoch 7**

| Epoch | Training Loss | Validation Loss |   Accuracy |
| ----: | ------------: | --------------: | ---------: |
|     1 |        2.4833 |          1.9019 |     90.53% |
|     2 |        1.1780 |          0.9235 |     94.26% |
|     3 |        0.5889 |          0.5585 |     95.70% |
|     4 |        0.2858 |          0.4286 |     95.84% |
|     5 |        0.3016 |          0.3949 |     95.84% |
|     6 |        0.2069 |          0.3780 |     95.84% |
|     7 |        0.1563 |          0.3599 | **96.56%** |
|     8 |        0.1414 |      **0.3530** |     96.27% |
|     9 |        0.1093 |          0.3711 |     96.27% |
|    10 |        0.1033 |          0.3635 |     96.27% |

> Performance reported here is based on the validation split. The notebook does not include an independent test set.

---

## 💻 Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Scikit-learn
* PIL
* NumPy
* Jupyter Notebook
* Google Colab
* Kaggle

### Main Components

```text
AutoImageProcessor
AutoModelForImageClassification
Trainer
TrainingArguments
```

---

## ▶️ Running the Project

The project is provided as a Jupyter Notebook:

```text
index.ipynb
```

The notebook downloads the Tobacco3482 dataset using the **Kaggle API**, prepares the images, fine-tunes the DiT model, and evaluates its validation performance.

The current notebook execution uses a **Google Colab environment**, with Kaggle dataset download commands and Google Drive access for the Kaggle API credentials.

---

## 🔬 Methodology

### 1. Dataset Loading

The Tobacco3482 dataset is downloaded and organized by document category.

### 2. Label Encoding

Each category is mapped to a numerical class ID.

### 3. Stratified Splitting

The data is divided into training and validation subsets while preserving class proportions.

### 4. Image Processing

Images are converted to RGB and processed using `AutoImageProcessor`.

### 5. Transfer Learning

A pretrained DiT model is loaded using the `microsoft/dit-base-finetuned-rvlcdip` checkpoint.

### 6. Fine-Tuning

The model is fine-tuned for the 10 Tobacco3482 categories.

### 7. Evaluation

Performance is measured using **validation accuracy** and **validation loss**.

---

## ⚠️ Current Limitations

* No independent test set is included.
* Evaluation currently focuses on accuracy and loss.
* No precision, recall, F1-score, or confusion matrix is reported.
* No per-class performance analysis is included.
* The notebook depends on environment-specific Kaggle and Google Drive configuration.

---

## 🔮 Future Improvements

* Add an independent test set.
* Add precision, recall, and F1-score.
* Generate a confusion matrix.
* Analyze per-class performance.
* Explore document-specific augmentation.
* Compare DiT with other vision architectures.
* Add inference on new document images.
* Improve experiment reproducibility.

---

## 📚 References

* Tobacco3482 Dataset
* Microsoft Document Image Transformer (DiT)
* Hugging Face Transformers
* PyTorch
* Scikit-learn
