# 📄 DocumentVision-DiT

### Document Image Classification with DiT and Transfer Learning

A deep learning project for **automatic document image classification** using a pretrained **Document Image Transformer (DiT)** model fine-tuned on the **Tobacco3482** dataset.

The system classifies document images into **10 predefined categories** using transfer learning and fine-tuning.

---

## 🎯 Project Overview

Document collections often contain different types of documents such as emails, forms, letters, reports, resumes, and scientific documents.

Manually organizing large collections can be time-consuming. This project explores an automated deep learning approach for recognizing the type of a document directly from its image.

The project uses a pretrained **Microsoft DiT model** and adapts its classification head to the 10 document categories available in the Tobacco3482 dataset.

### Main Pipeline

```text
Tobacco3482 Dataset
        │
        ▼
Document Images
        │
        ▼
Train / Validation Split
        │
        ▼
Image Preprocessing
(AutoImageProcessor)
        │
        ▼
Pretrained DiT
        │
        ▼
Fine-Tuning
        │
        ▼
10-Class Classification
        │
        ▼
Model Evaluation
```

---

## 🧠 Model

The project uses:

**Microsoft DiT — Document Image Transformer**

Pretrained model:

```text
microsoft/dit-base-finetuned-rvlcdip
```

The original pretrained model was adapted to the target dataset using:

* Transfer Learning
* Fine-Tuning
* A new classification head
* 10 target classes

The original classifier contains 16 output classes, while Tobacco3482 contains 10 classes in this project. Therefore, the classification layer is reinitialized to match the target number of classes.

---

## 📊 Dataset

The project uses the **Tobacco3482** document image dataset.

Dataset:

```text
patrickaudriaz/tobacco3482jpg
```

The dataset contains **3,482 document images** organized into 10 categories.

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

## 🗂️ Dataset Structure

The images are organized according to their class:

```text
Tobacco3482-jpg/
│
├── ADVE/
├── Email/
├── Form/
├── Letter/
├── Memo/
├── News/
├── Note/
├── Report/
├── Resume/
└── Scientific/
```

Each folder represents one document category.

---

## 🔄 Data Preparation

The project performs the following preparation steps:

1. Locate the dataset.
2. Collect image paths.
3. Generate numerical labels from folder names.
4. Map class names to class IDs.
5. Split the dataset into training and validation sets.
6. Use the model-specific `AutoImageProcessor` to prepare the images.

The split uses **stratified sampling** to preserve the distribution of document classes.

```python
train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)
```

The current implementation uses:

```text
80% Training
20% Validation
```

---

## ⚙️ Training Configuration

The main training configuration includes:

| Parameter              |                                  Value |
| ---------------------- | -------------------------------------: |
| Model                  | `microsoft/dit-base-finetuned-rvlcdip` |
| Dataset                |                            Tobacco3482 |
| Number of classes      |                                     10 |
| Train/Validation split |                                  80/20 |
| Batch size             |                                      8 |
| Learning rate          |                                 `2e-5` |
| Epochs                 |                                     10 |
| Weight decay           |                                 `0.01` |
| Random state           |                                     42 |
| Mixed precision        |            FP16 when CUDA is available |
| Framework              |    PyTorch / Hugging Face Transformers |

---

## 📈 Results

After fine-tuning for 10 epochs, the model achieved:

### **96.27% Validation Accuracy**

The best observed validation accuracy during training was approximately:

### **96.56%**

The final evaluation reported:

```text
Evaluation Loss:      0.353009
Validation Accuracy:  0.962697
```

### Training Progress

| Epoch | Training Loss | Validation Loss | Validation Accuracy |
| ----: | ------------: | --------------: | ------------------: |
|     1 |         2.483 |           1.902 |              90.53% |
|     2 |         1.178 |           0.924 |              94.26% |
|     3 |         0.589 |           0.558 |              95.70% |
|     4 |         0.286 |           0.429 |              95.84% |
|     5 |         0.302 |           0.395 |              95.84% |
|     6 |         0.207 |           0.378 |              95.84% |
|     7 |         0.156 |           0.360 |          **96.56%** |
|     8 |         0.141 |       **0.353** |              96.27% |
|     9 |         0.109 |           0.371 |              96.27% |
|    10 |         0.103 |           0.364 |              96.27% |

> **Note:** The reported performance is validation performance. The current implementation does not include an independent test set.

---

## 💻 Technologies

The project is implemented using Python and modern deep learning libraries.

### Main technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Scikit-learn
* PIL
* NumPy
* Matplotlib

### Main Hugging Face components

```python
AutoImageProcessor
AutoModelForImageClassification
Trainer
TrainingArguments
```

---

## 📁 Project Structure

A recommended repository structure is:

```text
DocumentVision-DiT/
│
├── README.md
├── notebook/
│   └── document_classification.ipynb
│
├── results/
│   ├── training_results/
│   └── figures/
│
├── requirements.txt
│
└── .gitignore
```

> The exact repository structure can be adjusted depending on whether the project is kept as a notebook-based experiment or converted into a modular Python project.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/DocumentVision-DiT.git
cd DocumentVision-DiT
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Example dependencies:

```text
torch
torchvision
transformers
datasets
scikit-learn
numpy
pillow
matplotlib
```

For GPU training, an appropriate CUDA-compatible PyTorch installation is recommended.

---

## ▶️ Running the Project

The project can be executed using the provided Jupyter Notebook.

```bash
jupyter notebook
```

Then open:

```text
document_classification.ipynb
```

The notebook performs:

```text
Dataset loading
      ↓
Class identification
      ↓
Train/Validation split
      ↓
Image preprocessing
      ↓
Model initialization
      ↓
Fine-tuning
      ↓
Evaluation
```

---

## 🔬 Methodology

### 1. Dataset Loading

The Tobacco3482 dataset is downloaded and organized according to document categories.

### 2. Label Encoding

Each document category is converted into a numerical class ID.

### 3. Stratified Splitting

The dataset is divided into training and validation subsets while preserving class proportions.

### 4. Image Processing

Images are processed using the image processor associated with the pretrained DiT model.

### 5. Transfer Learning

A pretrained DiT model is loaded to take advantage of previously learned visual representations.

### 6. Fine-Tuning

The model is fine-tuned on the Tobacco3482 dataset.

The final classification layer is adapted from the original 16-class configuration to the 10 target classes.

### 7. Evaluation

The model is evaluated using validation accuracy and validation loss.

---

## ⚠️ Current Limitations

The current version is a baseline implementation and has several limitations:

* No independent test set is currently used.
* Evaluation is mainly based on accuracy.
* Precision, recall, F1-score, and confusion matrix are not yet included.
* Per-class performance analysis is not currently reported.
* The current notebook contains some environment-specific dataset loading code.
* Further experiments are needed to assess generalization.

These limitations provide opportunities for future improvements.

---

## 🔮 Future Improvements

Possible extensions include:

* [ ] Add an independent test set
* [ ] Generate a classification report
* [ ] Calculate precision, recall, and F1-score
* [ ] Add a confusion matrix
* [ ] Analyze performance for each document category
* [ ] Add appropriate document-image augmentation
* [ ] Improve experiment reproducibility
* [ ] Compare DiT with other vision models
* [ ] Perform hyperparameter optimization
* [ ] Add inference on new document images
* [ ] Build a simple web interface for document classification

---

## 📌 Example Use Case

A user provides a document image:

```text
document.jpg
      │
      ▼
   DiT Model
      │
      ▼
Predicted Class
      │
      ▼
   "Resume"
```

The system can therefore be used as a starting point for automated document organization and classification.

---

## 📚 References

* Tobacco3482 document image dataset
* Microsoft Document Image Transformer (DiT)
* Hugging Face Transformers
* PyTorch
* Scikit-learn



---

## ⭐ Acknowledgment

This project was developed as an exploration of **Deep Learning, Vision Transformers, Transfer Learning, and Document Image Classification**.

If you find this project useful, feel free to ⭐ star the repository.
