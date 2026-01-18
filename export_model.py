import keras
import os

# 1. Load your Keras 3 model
print("Loading model...")
model = keras.models.load_model('models/my_model.keras')

# 2. Export it as a standard TensorFlow SavedModel (Folder)
# This removes the "Keras version" issues by saving the raw graph.
output_path = 'models/my_saved_model'
model.export(output_path)

print(f"Success! Model exported to folder: {output_path}")