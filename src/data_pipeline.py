import os
import zipfile
import requests
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Zenodo RAVDESS Audio only
DATA_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# We only want: 01 = neutral, 03 = happy, 04 = sad, 05 = angry
# RAVDESS filename format: 03-01-01-01-01-01-01.wav
# Emotion is the 3rd index (0-based: index 2) -> 01, 02, 03, 04, 05, 06, 07, 08
EMOTION_MAP = {
    "01": 0, # Neutral
    "03": 1, # Happy
    "04": 2, # Sad
    "05": 3  # Angry
}

def download_and_extract():
    """Downloads the RAVDESS dataset from Zenodo and extracts it."""
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ravdess.zip")
    
    # Check if we already have the extracted folders (Actor_01 to Actor_24)
    actor_1_path = os.path.join(RAW_DIR, "Actor_01")
    if os.path.exists(actor_1_path):
        print("Dataset already extracted. Skipping download.")
        return

    if not os.path.exists(zip_path):
        print(f"Downloading RAVDESS dataset from {DATA_URL}...")
        response = requests.get(DATA_URL, stream=True)
        response.raise_for_status()
        
        # We can implement a simple progress meter
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = int(50 * downloaded / total_size)
                    print(f"\r[{'=' * percent}{' ' * (50-percent)}] {downloaded}/{total_size} bytes", end="")
        print("\nDownload complete.")
    
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print("Error: Bad zip file. Deleting the corrupted zip file. Please run again.")
        os.remove(zip_path)
        exit(1)

def extract_mfcc(file_path):
    """Extracts MFCC features from an audio file."""
    # Load audio. RAVDESS files are about 3 seconds long.
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Average along the time axis
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

def process_data():
    """Iterates through extracted RAVDESS audio files, filters emotions, and extracts features."""
    X = []
    y = []
    
    print("Processing audio files and extracting MFCC features...")
    count = 0
    # Iterate through actors
    if not os.path.exists(RAW_DIR):
        print("Raw directory not found.")
        return X, y
        
    for actor_dir in os.listdir(RAW_DIR):
        actor_path = os.path.join(RAW_DIR, actor_dir)
        if not os.path.isdir(actor_path):
            continue
            
        for file in os.listdir(actor_path):
            if not file.endswith(".wav"):
                continue
                
            # Parse emotion from filename
            # Example: 03-01-05-01-01-01-01.wav
            parts = file.split("-")
            if len(parts) >= 3:
                emotion = parts[2]
                
                if emotion in EMOTION_MAP:
                    file_path = os.path.join(actor_path, file)
                    feature = extract_mfcc(file_path)
                    X.append(feature)
                    y.append(EMOTION_MAP[emotion])
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} files...")
                        
    return np.array(X), np.array(y)

def main():
    download_and_extract()
    
    X, y = process_data()
    
    if len(X) == 0:
        print("No valid data found or processed. Aborting.")
        return
        
    print(f"\nFinal count: Extracted {len(X)} samples.")
    
    # Convert labels to categorical (one-hot encoding)
    y_cat = to_categorical(y, num_classes=4)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)
    
    classes = np.array(["Neutral", "Happy", "Sad", "Angry"])
    np.save(os.path.join(PROCESSED_DIR, "classes.npy"), classes)
    print(f"Data processing complete. Files saved to {PROCESSED_DIR}/")

if __name__ == "__main__":
    main()
