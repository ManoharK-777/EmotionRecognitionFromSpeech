import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization

def build_model(input_shape, num_classes=4):
    """
    Builds a 1D Convolutional Neural Network (CNN) model for Speech Emotion Recognition 
    using extracted MFCC features.
    
    Args:
        input_shape (tuple): Shape of the input features (e.g., (40, 1))
        num_classes (int): Number of emotion classes to predict
        
    Returns:
        Compiled Keras model ready for training
    """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    # We use Adam optimizer and Categorical Crossentropy loss since classes are one-hot encoded
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
                  
    return model
