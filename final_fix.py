import sys
import types
import os

# --- STEP 1: THE TRICK ---
# We create a "fake" module in memory. 
# When tensorflowjs asks "Is tensorflow_decision_forests installed?", 
# Python will say "Yes" (even though it's empty).
mock_df = types.ModuleType("tensorflow_decision_forests")
sys.modules["tensorflow_decision_forests"] = mock_df

# --- STEP 2: IMPORT ---
# Now we can safely import the converter without it crashing
import tensorflowjs as tfjs
import tensorflow as tf

# --- STEP 3: CONVERT ---
input_path = 'models/my_model.h5' 
output_path = 'static/tfjs_model'

print(f"1. Loading model from {input_path}...")
# Load the H5 file you created earlier
model = tf.keras.models.load_model(input_path)

print(f"2. Converting to {output_path}...")
# Use the Python API directly (bypassing the broken command line)
tfjs.converters.save_keras_model(model, output_path)

print("✅ SUCCESS! Check your static/tfjs_model folder.")