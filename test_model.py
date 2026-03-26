import numpy as np
import tensorflow as tf
from src.model import build_model

def test_model_build():
    """Tests if the CNN model can be built and compiled correctly."""
    print("Testing model build...")
    try:
        # Dummy input shape: (40 MFCC coefficients, 1 channel)
        input_shape = (40, 1)
        model = build_model(input_shape, num_classes=4)
        
        # Print summary
        model.summary()
        print("\n✅ Model built and compiled successfully!")
        
        # Test a dummy prediction
        dummy_input = np.random.rand(1, 40, 1)
        output = model.predict(dummy_input)
        print(f"✅ Dummy prediction shape: {output.shape} (Expected: (1, 4))")
        
    except Exception as e:
        print(f"❌ Error building model: {e}")

if __name__ == "__main__":
    test_model_build()
