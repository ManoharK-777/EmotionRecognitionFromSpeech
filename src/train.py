import os
import numpy as np
import tensorflow as tf
from model import build_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "output"

def load_data():
    """Loads preprocessed datasets."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    classes = np.load(os.path.join(DATA_DIR, "classes.npy"))
    
    # Reshape X for Conv1D: (samples, time_steps, features)
    # Our extract_mfcc gives (40,) per sample, so reshape to (samples, 40, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    return X_train, X_test, y_train, y_test, classes

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plots and saves confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

def main():
    print("Loading preprocessed dataset...")
    try:
        X_train, X_test, y_train, y_test, classes = load_data()
    except FileNotFoundError:
        print("Dataset not found! Please run 'python src/data_pipeline.py' first.")
        return
        
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Classes: {classes}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, num_classes=len(classes))
    model.summary()
    
    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, "best_model.keras")
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_path, 
                                                    monitor='val_accuracy', 
                                                    save_best_only=True, 
                                                    mode='max', 
                                                    verbose=1)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                      patience=15, 
                                                      restore_best_weights=True)
                                                      
    # Train
    print("\nTraining model...")
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=100, 
                        batch_size=32, 
                        callbacks=[checkpoint, early_stopping])
    
    # Predict and Evaluate
    print("\nEvaluating model...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write("SPEECH EMOTION RECOGNITION - CLASSIFICATION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(report)
        
    plot_confusion_matrix(y_true, y_pred, classes)
    print(f"\nTraining completed!")
    print(f"- Best model saved to: {best_model_path}")
    print(f"- Evaluation metrics saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
