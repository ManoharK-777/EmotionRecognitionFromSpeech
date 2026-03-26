# 🎙️ Speech Emotion Recognition System

An end-to-end Machine Learning web application to detect human emotions (Happy, Sad, Angry, Neutral) from speech audio using Deep Learning (CNN). Built with Python, TensorFlow/Keras, Librosa, and a modern Streamlit frontend.

---

## 🎯 Features
- **Data Preprocessing Script**: Automatically downloads the RAVDESS dataset and extracts MFCC features.
- **Deep Learning Model**: A 1D Convolutional Neural Network (CNN) built in TensorFlow/Keras optimized for Audio analysis.
- **Modern Dashboard**: Clean, responsive frontend built with Streamlit and styled with custom CSS.
- **Audio Analysis**: Visualizes audio waveforms and MFCC spectrograms.
- **Real-time Prediction**: Upload `.wav` files and see the predicted emotion with confidence scores.

## 🛠️ Project Structure
```text
SpeechEmotionRecognition/
│
├── data/
│   ├── raw/                 # Downloaded raw RAVDESS audio dataset
│   └── processed/           # Extracted MFCC features (.npy files)
│
├── models/
│   └── best_model.keras     # Trained CNN model
│
├── output/
│   ├── classification_report.txt  # Evaluation metrics
│   └── confusion_matrix.png       # Confusion matrix visualization
│
├── src/
│   ├── data_pipeline.py     # Download & Extract data, preprocess MFCCs
│   ├── model.py             # Defines the CNN architecture
│   └── train.py             # Script to train and save the model
│
├── app.py                   # Streamlit Frontend Application
├── requirements.txt         # Project dependencies
└── README.md                # Documentation (this file)
```

## 🚀 Getting Started

### 1. Install Dependencies
Make sure you have Python 3.9+ installed.
```shell
pip install -r requirements.txt
```

### 2. Prepare the Data
Run the data pipeline to automatically download the RAVDESS dataset, extract it, and process the MFCC features. *(Note: The download is around ~400MB and might take some time.)*
```shell
python src/data_pipeline.py
```

### 3. Train the Model
Once the data is processed, run the training script. This will evaluate the model and save `best_model.keras` to the `models/` directory.
```shell
python src/train.py
```

### 4. Run the Web Application
Start the Streamlit dashboard:
```shell
streamlit run app.py
```
Open the provided local URL (e.g., `http://localhost:8501`) in your browser. Upload any `.wav` file to detect the emotion!

## 🤝 Dataset Credit
This project uses the **[RAVDESS](https://zenodo.org/record/1188976)** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset by Livingstone & Russo (2018).

Enjoy analyzing speech emotions! 🎉
