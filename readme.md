````markdown name=README.md
# 3D CNN Framework for Alzheimer's Disease Classification

A deep-learning framework for classifying Alzheimer's disease (AD) and cognitively normal (CN) subjects using 3D brain MRI hippocampal data.

This project preprocesses MRI scans, removes non-brain tissue, extracts the left and right hippocampus, generates normalized 3D patches, and classifies subjects using a dual-stream 3D Convolutional Neural Network.

> **Disclaimer:** This project is intended for research and educational purposes only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

---

## Project Overview

The system follows this workflow:

```text
MRI scans
   │
   ├── Skull stripping
   │
   ├── Hippocampus segmentation
   │
   ├── Left/right hippocampus extraction
   │
   ├── 3D patch cropping and normalization
   │
   ├── Subject-level train/validation/test splitting
   │
   └── Dual-stream 3D CNN classification
           ├── Left hippocampus branch
           ├── Right hippocampus branch
           └── Feature fusion and AD/CN prediction
```

For every subject, the model receives two inputs:

- A left hippocampus 3D patch.
- A right hippocampus 3D patch.

The two inputs are processed independently by separate 3D CNN branches. Their extracted features are then combined to produce the final classification.

---

## Main Objective

The objective of this project is to investigate whether structural information from the left and right hippocampus in MRI scans can be used to distinguish between:

- **AD** — Alzheimer's disease
- **CN** — Cognitively normal subjects

The final model produces an Alzheimer's disease probability using a sigmoid output layer.

---

## Repository Structure

```text
.
├── hippocampus_segmentation.ipynb
├── model_training.ipynb
├── final_fused_model.keras
│
├── Preprocessing/
│   ├── load_data.ipynb
│   ├── skull_strip_all.py
│   ├── segmentation.py
│   ├── crop_normalize_patches.py
│   └── data_splitter.py
│
└── Other saved models/
    ├── best_dual_model.keras
    ├── best_fuse_model.keras
    ├── best_model_fold_1.keras
    ├── best_model_fold_2.keras
    ├── best_model_fold_3.keras
    ├── best_model_fold_4.keras
    └── best_model_fold_5.keras
```

---

## File Descriptions

### `Preprocessing/load_data.ipynb`

This notebook is used for inspecting and loading the original MRI data.

It:

- Loads NIfTI MRI files using `nibabel`.
- Reads MRI scans from the `AD` and `CN` directories.
- Assigns labels to the images:
  - `1` represents Alzheimer's disease.
  - `0` represents cognitively normal subjects.
- Displays image shape, voxel spacing, data type, and dataset statistics.
- Prints the file paths used in the dataset.

This notebook helps verify that the original MRI data has been loaded correctly.

---

### `Preprocessing/skull_strip_all.py`

This script performs skull stripping on the MRI scans using the `hd-bet` tool.

It:

- Searches for MRI files inside:
  - `data/AD`
  - `data/CN`
- Creates an `output_brain` directory.
- Processes `.nii` and `.nii.gz` files.
- Removes non-brain tissue from each MRI scan.
- Saves the processed scans with the `_brain.nii.gz` suffix.

Skull stripping reduces irrelevant information and allows the later stages to focus on brain structures.

The script uses the following command:

```bash
hd-bet -i input_file -o output_file -device cpu
```

---

### `Preprocessing/segmentation.py`

This script extracts the left and right hippocampus from the skull-stripped brain images.

It:

- Loads skull-stripped brain images.
- Finds the corresponding left and right hippocampus masks.
- Supports multiple mask naming conventions.
- Binarizes the masks.
- Applies the masks to the brain images.
- Produces separate left and right hippocampus volumes.
- Saves the extracted hippocampus images as NIfTI files.
- Reports missing masks and image shape mismatches.

The resulting files are saved in a structure similar to:

```text
hippo_only/
├── AD/
│   ├── subject_left_hippo.nii.gz
│   └── subject_right_hippo.nii.gz
└── CN/
    ├── subject_left_hippo.nii.gz
    └── subject_right_hippo.nii.gz
```

---

### `Preprocessing/crop_normalize_patches.py`

This script converts hippocampus volumes into fixed-size 3D patches that can be used as CNN inputs.

It:

- Loads left and right hippocampus NIfTI files.
- Finds the center of the non-zero hippocampus region.
- Crops a `64 × 64 × 64` voxel patch.
- Adds zero padding when necessary.
- Applies z-score normalization.
- Adds a channel dimension.
- Saves the patches as NumPy `.npy` files.

The generated patch shape is:

```text
(1, 64, 64, 64)
```

The output directory is organized as follows:

```text
patches_out/
├── left/
│   ├── AD/
│   └── CN/
└── right/
    ├── AD/
    └── CN/
```

The normalization is calculated mainly from non-zero voxels so that background values do not dominate the intensity statistics.

---

### `Preprocessing/data_splitter.py`

This script divides the processed data into training, validation, and test sets.

It:

- Extracts the subject ID from each file name.
- Matches left and right hippocampus patches from the same subject.
- Ignores subjects for whom either the left or right patch is missing.
- Splits the subjects into:
  - 70% training data.
  - 15% validation data.
  - 15% test data.
- Uses a fixed random seed of `42`.
- Copies paired left/right patches into the split directories.

The resulting structure is:

```text
split_subject/
├── left/
│   ├── train/
│   │   ├── AD/
│   │   └── CN/
│   ├── val/
│   │   ├── AD/
│   │   └── CN/
│   └── test/
│       ├── AD/
│       └── CN/
│
└── right/
    ├── train/
    │   ├── AD/
    │   └── CN/
    ├── val/
    │   ├── AD/
    │   └── CN/
    └── test/
        ├── AD/
        └── CN/
```

Splitting at the subject level helps prevent images from the same subject from appearing in multiple subsets.

---

### `hippocampus_segmentation.ipynb`

This notebook is used for experimenting with or inspecting the hippocampus segmentation process.

It can be used to:

- Investigate segmentation outputs.
- Inspect hippocampus regions.
- Test segmentation-related processing steps.
- Verify that left and right hippocampus regions are correctly extracted.

The exact workflow may depend on the segmentation masks and data available in the local environment.

---

### `model_training.ipynb`

This is the main notebook for model training, evaluation, and prediction.

It performs the following tasks:

1. Loads paired left and right hippocampus `.npy` files.
2. Assigns labels for AD and CN subjects.
3. Builds TensorFlow datasets.
4. Batches and prefetches the data.
5. Applies 3D data augmentation.
6. Builds the dual-stream 3D CNN.
7. Compiles the model.
8. Applies class weights to address class imbalance.
9. Trains the model.
10. Saves the best model checkpoint.
11. Plots accuracy, AUC, and loss.
12. Evaluates the model on the test set.
13. Generates precision, recall, F1-score, and confusion matrix results.
14. Demonstrates prediction on a new subject.

The notebook uses:

- TensorFlow/Keras
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

---

### `final_fused_model.keras`

This is the final saved Keras model generated by the training notebook.

It contains:

- The dual-stream model architecture.
- The learned model weights.
- The left hippocampus processing branch.
- The right hippocampus processing branch.
- The final feature-fusion classification layer.

This model can be loaded for inference using TensorFlow/Keras.

---

### `Other saved models/`

This directory contains additional saved model checkpoints and model variants.

#### `best_dual_model.keras`

A saved checkpoint of the dual-input model that processes the left and right hippocampus independently.

#### `best_fuse_model.keras`

A saved fusion-model variant that combines features from the two hippocampus branches.

#### `best_model_fold_1.keras` to `best_model_fold_5.keras`

Saved models from different folds or training runs. These files can be used for:

- Comparing different experiments.
- Restoring earlier model checkpoints.
- Testing fold-specific performance.
- Building an ensemble model.

The exact differences between these models should be confirmed from the corresponding training experiments.

---

## Model Architecture

The model uses two independent 3D CNN branches:

```text
Left hippocampus
        │
        ▼
   Left 3D CNN
        │
        ▼
 Left feature vector
        │
        ├──────────────┐
        │              │
        │              ▼
        │        Feature fusion
        │              │
        │              ▼
        │       Binary classifier
        │              │
        │              ▼
        │        AD probability
        │
Right hippocampus
        │
        ▼
  Right 3D CNN
        │
        ▼
Right feature vector
```

Each CNN branch contains:

- 3D convolutional layers.
- Batch normalization.
- ReLU activation.
- 3D max-pooling layers.
- Global average pooling.
- Dense layers.
- Dropout regularization.

The left and right branches have separate weights. Their feature vectors are concatenated before being passed to the final sigmoid classification layer.

---

## Input Format

The model expects two 3D inputs:

```text
Left input:  64 × 64 × 64 × 1
Right input: 64 × 64 × 64 × 1
```

The class labels are:

```text
0 = CN
1 = AD
```

The model returns a probability between `0` and `1`.

For example:

```text
Probability: 0.82
Prediction: AD
```

A threshold of `0.5` is used in the notebook:

```python
prediction = "AD" if probability >= 0.5 else "CN"
```

---

## Requirements

The project requires Python 3 and the following packages:

```text
numpy
tensorflow
nibabel
scikit-learn
matplotlib
seaborn
jupyter
hd-bet
```

You can install the Python dependencies with:

```bash
pip install numpy tensorflow nibabel scikit-learn matplotlib seaborn jupyter
```

Install `hd-bet` according to its official installation instructions.

A GPU is recommended for training because 3D CNN models require considerable computational resources.

---

## Data Preparation Workflow

The scripts should generally be executed in the following order:

### 1. Load and inspect the data

Open:

```text
Preprocessing/load_data.ipynb
```

Use this notebook to verify the original MRI files and class labels.

### 2. Perform skull stripping

```bash
python Preprocessing/skull_strip_all.py
```

This creates skull-stripped brain images.

### 3. Extract the hippocampus

```bash
python Preprocessing/segmentation.py
```

This uses segmentation masks to create separate left and right hippocampus volumes.

### 4. Generate normalized patches

```bash
python Preprocessing/crop_normalize_patches.py
```

This creates normalized `64 × 64 × 64` NumPy patches.

### 5. Create train, validation, and test splits

```bash
python Preprocessing/data_splitter.py
```

This creates paired subject-level splits.

### 6. Train the model

Open:

```text
model_training.ipynb
```

Update the dataset paths and run the notebook cells in order.

---

## Important Configuration

The preprocessing scripts currently contain hard-coded local paths, such as:

```python
C:\Users\chama\OneDrive\Desktop\research
```

These paths must be updated before running the project on another computer.

For example, update variables such as:

```python
LEFT_ROOT
RIGHT_ROOT
OUT_ROOT
BRAIN_ROOT
MASK_ROOT
PATCHES_ROOT
SPLIT_ROOT
```

Also update the dataset path in `model_training.ipynb`:

```python
SPLIT_ROOT = "path/to/split_subject"
```

---

## Loading the Trained Model

The trained model can be loaded using TensorFlow/Keras:

```python
import numpy as np
from tensorflow import keras

model = keras.models.load_model("final_fused_model.keras")

left = np.load("path/to/subject_left.npy")
right = np.load("path/to/subject_right.npy")

# Convert channel-first patches to Keras input format.
left = np.transpose(left, (1, 2, 3, 0))
right = np.transpose(right, (1, 2, 3, 0))

# Add batch dimension.
left = left[None, ...].astype("float32")
right = right[None, ...].astype("float32")

probability = float(
    model.predict([left, right], verbose=0)[0][0]
)

prediction = "AD" if probability >= 0.5 else "CN"

print("Prediction:", prediction)
print("AD probability:", probability)
```

The input data must use the same preprocessing and normalization process used during training.

---

## Example Evaluation Results

The training notebook includes an example evaluation from one experiment:

```text
Accuracy:  0.8333
AUC:       0.9429
Precision: 0.7143
Recall:    1.0000
F1 Score:  0.8333
```

These results depend on the dataset, split, preprocessing, random seed, and training configuration. They should not be considered representative of clinical performance.

---

## Reproducibility

The project uses a random seed of `42` in several places:

- NumPy.
- TensorFlow.
- Data splitting.

However, complete reproducibility may still depend on:

- TensorFlow version.
- CUDA and cuDNN versions.
- Hardware.
- Dataset ordering.
- Operating system.
- File naming and directory structure.

---

## Limitations

- The dataset is relatively small.
- The model may overfit to the available subjects.
- The reported metrics may vary significantly between data splits.
- The preprocessing scripts use hard-coded local paths.
- The scripts depend on specific file naming conventions.
- The dataset is not included in this repository.
- The project does not currently provide an automated end-to-end pipeline.
- The model has not been clinically validated.
- Performance on other datasets or populations may be different.

---

## Medical Disclaimer

This project is intended only for research and educational purposes.

It is not intended to:

- Diagnose Alzheimer's disease.
- Replace a medical professional.
- Provide medical advice.
- Recommend treatment.
- Make clinical decisions.

Any clinical use would require extensive validation, expert review, regulatory approval, privacy protections, and testing on large and diverse patient populations.

---

## License

No license is currently specified for this repository.

If you plan to publish, reuse, or distribute this project, consider adding an appropriate open-source license. Also verify that the original MRI data, segmentation masks, and trained model files can legally be redistributed.

````
