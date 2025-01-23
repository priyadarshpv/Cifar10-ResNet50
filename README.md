# CIFAR-10 Image Classification with ResNet50

## Project Overview
This project focuses on building an image classification model for the CIFAR-10 dataset using a ResNet50 architecture. The CIFAR-10 dataset consists of 60,000 32x32 RGB images across 10 classes. The model employs TensorFlow and Keras for training and evaluation and utilizes transfer learning via ResNet50 for better performance on the classification task.

---

## Dataset Details
- **Dataset Source:** [Kaggle CIFAR-10 competition](https://www.kaggle.com/c/cifar-10)
- **Number of Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Number of Images:**
  - Training Set: 50,000 images
  - Test Set: 10,000 images

---

## Project Workflow

### 1. Dataset Preparation
1. **Download CIFAR-10 dataset:**
   ```bash
   !pip install kaggle
   !mkdir -p ~/.config/kaggle
   !mv /content/kaggle.json ~/.config/kaggle/
   !chmod 600 ~/.config/kaggle/kaggle.json
   !kaggle competitions download -c cifar-10
   !unzip /content/cifar-10.zip
   ```

2. **Extract Training Images:**
   ```python
   import py7zr

   archive = py7zr.SevenZipFile('/content/train.7z', mode='r')
   archive.extractall()
   archive.close()
   ```

3. **Label Processing:**
   - Convert class names to numerical labels for compatibility with the neural network.

4. **Image Processing:**
   - Convert images to numpy arrays.
   - Normalize pixel values by scaling them between 0 and 1.

### 2. Model Building

#### Baseline Neural Network
A simple feedforward neural network for baseline evaluation:
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_of_classes, activation='softmax')
])
```

#### ResNet50 Transfer Learning
Using ResNet50 pre-trained on ImageNet:
1. Load ResNet50 as the convolutional base (without the top layer):
   ```python
   convolutional_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
   ```
2. Add custom dense layers for classification:
   ```python
   model = models.Sequential([
       layers.UpSampling2D((2, 2)),
       layers.UpSampling2D((2, 2)),
       layers.UpSampling2D((2, 2)),
       convolutional_base,
       layers.Flatten(),
       layers.BatchNormalization(),
       layers.Dense(256, activation='relu'),
       layers.Dropout(0.5),
       layers.BatchNormalization(),
       layers.Dense(128, activation='relu'),
       layers.Dropout(0.5),
       layers.BatchNormalization(),
       layers.Dense(num_of_classes, activation='softmax')
   ])
   ```

3. Compile and Train:
   ```python
   model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-5),
                 loss='sparse_categorical_crossentropy',
                 metrics=['acc'])

   history = model.fit(x_train, y_train, validation_split=0.1, epochs=10)
   ```

### 3. Evaluation
- Evaluate the model on the test set:
  ```python
  model.evaluate(x_test, y_test)
  ```

- Generate classification report and confusion matrix:
  ```python
  from sklearn.metrics import confusion_matrix, classification_report

  y_pred = model.predict(x_test)
  y_pred_classes = np.argmax(y_pred, axis=1)

  print(confusion_matrix(y_test, y_pred_classes))
  print(classification_report(y_test, y_pred_classes))
  ```

### 4. Save and Load Model
Save the trained model for future use:
```python
model.save("cifar10_resnet50_model.h5")
model = load_model("cifar10_resnet50_model.h5")
```

---

## Results
### Model Performance:
| Metric         | Value        |
|----------------|--------------|
| **Accuracy**   | ~93%         |
| **F1-Score**   | 0.93         |

### Confusion Matrix:
A heatmap visualization was generated to show the performance across all classes.

---

## Visualizations
1. **Training and Validation Loss:**
   - Shows the reduction in loss over epochs.
2. **Training and Validation Accuracy:**
   - Demonstrates the improvement in accuracy over epochs.

---

## Installation and Requirements
### Prerequisites:
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### `requirements.txt`
```plaintext
tensorflow==2.11.0
numpy==1.23.5
pandas==1.5.2
matplotlib==3.7.1
seaborn==0.12.2
h5py==3.8.0
scikit-learn==1.2.0
Pillow==9.5.0
py7zr==0.20.4
```

---

## How to Run
1. Clone the repository.
2. Add your Kaggle API key (`kaggle.json`) to the working directory.
3. Download the dataset using Kaggle CLI.
4. Preprocess the data and run the ResNet50 model training script.
5. Evaluate the model and visualize the results.

---

## Project Structure
```
project/
|-- train/                # Contains training images
|-- trainLabels.csv       # Contains image labels
|-- cifar10_resnet50_model.h5   # Saved model
|-- main.py               # Script to preprocess, train, and evaluate
|-- requirements.txt      # Dependencies
|-- README.md             # Project documentation
```

---

## Future Work
1. **Data Augmentation:** To further improve model generalization.
2. **Hyperparameter Tuning:** Experiment with learning rates, optimizers, and dropout rates.
3. **Advanced Architectures:** Test models like EfficientNet or Vision Transformers.

---

## Acknowledgments
- **Kaggle:** For providing the CIFAR-10 dataset.
- **TensorFlow/Keras:** For model building and training.

---

## License
This project is licensed under the MIT License.


