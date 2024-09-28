# Product Classifier

This repository contains the code and resources for the **Product Classifier** project, which classifies different types of products using machine learning techniques. This project is implemented using Deep Learning models, including Convolutional Neural Networks (CNN), and leverages Python libraries such as TensorFlow and Keras.

## Overview

The Product Classifier project aims to classify various products based on their images. This is especially useful for e-commerce platforms to automatically categorize items. The model achieves high accuracy by using data augmentation techniques to increase the robustness and generalizability of the classification system.

## Features

- **Image Classification**: Classifies product images into predefined categories.
- **Data Augmentation**: Enhances the training dataset with transformations such as rotation, flipping, and scaling.
- **Model Evaluation**: Provides metrics such as accuracy and loss to evaluate model performance.
- **Pre-trained Model Usage**: Utilizes transfer learning techniques to improve model efficiency and accuracy.

## Google Drive Resources

The Google Drive folder contains the datasets, pre-trained models, and additional resources used in this project. You can access it using the following link:

[Google Drive Folder](https://drive.google.com/drive/folders/1Ipr6qs-_vHd7sq5YMgqef-Y2N3duTPkB?usp=sharing)

## Project Structure

- **data/**: Contains the dataset used for training and testing.
- **models/**: Pre-trained models and saved weights.
- **notebooks/**: Jupyter notebooks for experimentation and model training.
- **src/**: Source code for data preprocessing, model training, and evaluation.
- **README.md**: Project documentation.

## Getting Started

### Prerequisites

Ensure you have the following software and libraries installed:

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- pandas
- scikit-learn
- Matplotlib
- OpenCV

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/ShrivastavaPrakash/Products_classifer
    cd product-classifier
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and pre-trained models from the [Google Drive Folder](https://drive.google.com/drive/folders/1Ipr6qs-_vHd7sq5YMgqef-Y2N3duTPkB?usp=sharing) and place them in the respective directories.

### Usage

1. **Data Preprocessing**: Prepare the data by running the following script:

    ```bash
    python src/data_preprocessing.py
    ```

2. **Model Training**: Train the model using the following command:

    ```bash
    python src/train_model.py
    ```

3. **Model Evaluation**: Evaluate the trained model using:

    ```bash
    python src/evaluate_model.py
    ```

4. **Prediction**: Use the model to classify new product images:

    ```bash
    python src/predict.py --image_path <path_to_image>
    ```

## Results

The classifier achieved an accuracy of 92% on the test dataset. The model's performance can be further improved with additional data and fine-tuning.

## Contributing

If you would like to contribute to this project, please create a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For any questions or suggestions, feel free to contact me at [iampk9430@gmail.com](mailto:iampk9430@gmail.com).
