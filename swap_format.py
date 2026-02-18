import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

# 1. Load your modern Keras 3 model
print("Loading Keras 3 model...")
model = keras.models.load_model('models/my_model.keras')

# 2. Save it as the "Old School" H5 format
# This format is readable by EVERYTHING (Old and New tools)
print("Saving as Legacy H5...")
model.save('models/my_model.h5')

print("✅ Success! Created models/my_model.h5")