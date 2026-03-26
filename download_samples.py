import os
import requests
import zipfile

# EMO-DB Emotions: 
# W = Anger, F = Happiness, T = Sadness, N = Neutral
URL = "http://emodb.bilderbar.info/download/download.zip"
ZIP_PATH = "emodb.zip"
TARGET_DIR = "test_samples"

def main():
    print("Downloading 38MB EMO-DB sample dataset for Happy, Sad, Angry, Neutral...")
    response = requests.get(URL, stream=True)
    with open(ZIP_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                
    print("Download complete. Extracting 4 test files...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # We will pick one file for each emotion
    # The EMO-DB format uses the 6th character for emotion (index 5)
    # E.g., 03a01Fa.wav (F = Happiness)
    found = {"W": False, "F": False, "T": False, "N": False}
    mapping = {"W": "Angry", "F": "Happy", "T": "Sad", "N": "Neutral"}
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        for file in zf.namelist():
            if not file.endswith(".wav"): continue
            
            emotion_char = file[5]
            if emotion_char in found and not found[emotion_char]:
                # Extract and rename
                extracted_path = zf.extract(file, path=".")
                new_name = os.path.join(TARGET_DIR, f"{mapping[emotion_char]}_test_audio.wav")
                os.rename(extracted_path, new_name)
                found[emotion_char] = True
                print(f"Extracted: {new_name}")
                
            if all(found.values()):
                break
                
    # Clean up zip and remaining extracted wav directory (usually extracted to 'wav/')
    os.remove(ZIP_PATH)
    if os.path.exists("wav"):
        for root, dirs, files in os.walk("wav", topdown=False):
            for f in files: os.remove(os.path.join(root, f))
            os.rmdir(root)
            
    print("Successfully downloaded and prepared 4 test audio files in 'test_samples/'!")

if __name__ == "__main__":
    main()
