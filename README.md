# Image Classification Using ResNet50

## Overview
This project implements an image classification model using **ResNet50**, a popular pre-trained Convolutional Neural Network (CNN) architecture. The model is trained to classify images into 10 classes using a custom dataset. The project incorporates advanced techniques such as transfer learning, data preprocessing, and fine-tuning to achieve high accuracy and robust performance.

## Key Features
- Utilizes the **ResNet50** pre-trained model for feature extraction.
- Incorporates data augmentation for improved generalization.
- Implements transfer learning by freezing early layers of ResNet50.
- Includes dropout layers and batch normalization for regularization.
- Trains and evaluates the model with metrics such as accuracy and loss.

---

## Dataset
The dataset contains images classified into 10 categories. Each image is an RGB image with dimensions `(256, 256, 3)`.

### Dataset Preprocessing:
1. Resized all images to `(256, 256, 3)`.
2. Normalized pixel values to the range `[0, 1]`.
3. Applied data augmentation techniques, such as rotation, flipping, and zoom.

---

## Model Architecture
The model consists of the following components:
1. **ResNet50** (Pre-trained): Used as a feature extractor.
2. **Fully Connected Layers**: Added dense layers with:
   - Batch Normalization
   - Dropout (0.5 for regularization)
3. **Output Layer**: A dense layer with 10 units and a `softmax` activation function.

### Model Summary
- **Input Shape**: `(256, 256, 3)`
- **Base Model**: ResNet50 with frozen convolutional layers.
- **Custom Layers**:
  - Dense layers with ReLU activation
  - Dropout for regularization
  - Batch Normalization
- **Output Layer**: Dense layer with 10 neurons (softmax activation).

### Optimizer and Loss Function
- **Optimizer**: RMSprop with a learning rate of `2e-5`
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

---

## Results
| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |
|-------|--------------------|----------------------|-----------------|
| 1     | 31.04%            | 69.73%              | 0.9832          |
| 2     | 65.78%            | 84.47%              | 0.5293          |
| 3     | 77.65%            | 88.60%              | 0.3986          |
| 4     | 84.57%            | 90.38%              | 0.3291          |
| 5     | 88.89%            | 90.93%              | 0.3037          |
| 10    | 97.84%            | 92.73%              | 0.2443          |

### Classification Report
- **Overall Accuracy**: 93%
- **Precision, Recall, and F1-Score**:
  - Best-performing classes: `1`, `9`
  - Underperforming classes: `3`, `5`

---

## How to Run the Project

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Place your dataset in the `data/` folder.
   - Ensure images are organized into subfolders corresponding to class labels.

4. Train the model:
   ```bash
   python train.py
   ```

5. Evaluate the model:
   ```bash
   python evaluate.py
   ```

---

## Improvements and Future Work
1. **Augmentation**: Enhance the dataset using more augmentation techniques.
2. **Model Optimization**: Use advanced optimizers like AdamW or SGD with momentum.
3. **Ensemble Learning**: Combine predictions from multiple models.
4. **Hyperparameter Tuning**: Experiment with different dropout rates, learning rates, and batch sizes.
5. **Class Imbalance Handling**: Use class weighting or oversampling to improve performance on underperforming classes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For any queries, reach out to:
- **Name**: Priyadarsh
- **Email**: [priyadarshdinesh@gmail.com]

---

Happy Coding!

