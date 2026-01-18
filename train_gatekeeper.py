import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
DATA_DIR = 'gatekeeper_dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.90  # Stricter threshold for the final check

# --- 1. DATA GENERATORS (The "Fix" for Overfitting) ---
print("--- LOADING DATA ---")

# Generator 1: Training Data (Aggressive Augmentation)
# This makes the model see "hard" versions of images so it learns better.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,      
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.4,         
    horizontal_flip=True,
    vertical_flip=True,     
    fill_mode='nearest',
    validation_split=0.2
)

# Generator 2: Validation Data (Clean - NO Augmentation)
# This ensures we test the model on "real" images, not twisted ones.
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# Use val_datagen here
validation_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False 
)

# --- 2. SETUP CALLBACKS (The "Safety Brakes") ---
# 1. Checkpoint: Saves the model ONLY if it gets smarter (val_loss goes down).
checkpoint = ModelCheckpoint(
    'models/gatekeeper_model.keras', 
    monitor='val_loss', 
    save_best_only=True, 
    mode='min', 
    verbose=1
)

# 2. EarlyStopping: Quits if the model stops improving for 5 epochs.
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# --- 3. BUILD MODEL ---
print("\n--- BUILDING MODEL ---")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)  
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# --- 4. PHASE 1 TRAINING ---
print("\n--- PHASE 1: TRAINING HEAD ---")
history = model.fit(
    train_generator,
    epochs=10, 
    validation_data=validation_generator,
    callbacks=[checkpoint] # Save best model during Phase 1
)

# --- 5. PHASE 2: FINE-TUNING ---
print("\n--- PHASE 2: FINE-TUNING ---")

base_model.trainable = True
fine_tune_at = 120
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile with low learning rate
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train with both safety brakes enabled
history_fine = model.fit(
    train_generator,
    epochs=20, 
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop] 
)

# --- 6. EVALUATE BEST MODEL ---
# Important: Load the BEST saved version, not the last one in memory
print("\n--- LOADING BEST SAVED MODEL FOR EVALUATION ---")
best_model = load_model('models/gatekeeper_model.keras')

print(f"\n--- STRICT EVALUATION (Threshold: {CONFIDENCE_THRESHOLD}) ---")
probabilities = best_model.predict(validation_generator)
true_classes = validation_generator.classes
predicted_classes = (probabilities > CONFIDENCE_THRESHOLD).astype(int)

class_labels = list(validation_generator.class_indices.keys()) 
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(f"Refused Valid Plants (False Negatives): {cm[1][0]}")
print(f"Accepted Junk (False Positives): {cm[0][1]}")