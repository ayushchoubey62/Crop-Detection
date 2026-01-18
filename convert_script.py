import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Force TF backend for compatibility
import keras

# 1. Load your current model
print("Loading model...")
model = keras.models.load_model('models/my_model.keras')

# 2. Save it strictly as Legacy H5
# The 'save_format' flag is the key fix here!
print("Saving as Legacy H5...")
model.save('models/my_model.h5', save_format='h5')

print("Done! Ready for conversion.")