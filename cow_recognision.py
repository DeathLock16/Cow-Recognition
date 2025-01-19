#coding=Windows-1250
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import cv2
import numpy as np
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.src.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Load and preprocess the dataset
def load_images(dataset_dir, image_size=(128, 128)):
    images = []
    labels = []
    cow_classes = os.listdir(dataset_dir)

    for cow_class in cow_classes:
        cow_dir = os.path.join(dataset_dir, cow_class)
        if not os.path.isdir(cow_dir):
            continue

        for img_file in os.listdir(cow_dir):
            img_path = os.path.join(cow_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(cow_class)

    images = np.array(images, dtype="float32") / 255.0  # Normalize images
    labels = np.array(labels)
    return images, labels

# 2. Build a CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Load dataset and preprocess
dataset_dir = "database"  # Set your dataset directory here
image_size = (128, 128)

print("Loading dataset...")
images, labels = load_images(dataset_dir, image_size=image_size)

# Map labels to integers
label_to_index = {label: idx for idx, label in enumerate(np.unique(labels))}
index_to_label = {idx: label for label, idx in label_to_index.items()}
labels = np.array([label_to_index[label] for label in labels])

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 4. Train the model
print("Creating and training the model...")
input_shape = (image_size[0], image_size[1], 3)
num_classes = len(label_to_index)

model = create_model(input_shape, num_classes)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
datagen.fit(X_train)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=15,
                    verbose=1,
                    callbacks=[early_stopping])

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Dokładność - Zbiór treningowy', color='blue', linestyle='--')
plt.plot(history.history['val_accuracy'], label='Dokładność - Zbiór walidacyjny', color='green', linestyle='-')
plt.title('Porównanie dokładności modelu podczas treningu i walidacji', fontsize=14)
plt.xlabel('Epoki', fontsize=12)
plt.ylabel('Dokładność (%)', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid(visible=True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# 5. Evaluate the model
print("Evaluating the model...")

y_pred = np.argmax(model.predict(X_test), axis=1)

unique_classes_y_test = np.unique(y_test)
unique_classes_y_pred = np.unique(y_pred)

target_names = [index_to_label[i] for i in unique_classes_y_test]

if len(unique_classes_y_test) != len(unique_classes_y_pred):
    print(f"Warning: Number of unique classes in y_test ({len(unique_classes_y_test)}) does not match y_pred ({len(unique_classes_y_pred)})")

labels = unique_classes_y_test

print(classification_report(y_test, y_pred, target_names=target_names, labels=labels))


# 6. Save the model
model.save("cow_recognition_model.keras")
print("Model saved as cow_recognition_model.keras")

# 7. Load and predict a new image
def predict_cow(image_path, model, index_to_label, image_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, image_size)
        img = np.expand_dims(img, axis=0) / 255.0
        pred_idx = np.argmax(model.predict(img), axis=-1)[0]
        return index_to_label.get(pred_idx, "Unknown")
    return "Invalid image"

# Example usage
# Load the saved model
model = tf.keras.models.load_model("cow_recognition_model.keras")

# Predict a new cow image
probes = ["probes/a.jpg", "probes/b.jpg", "probes/c.jpg"]
predicted_cow = [predict_cow(probes[0], model, index_to_label), predict_cow(probes[1], model, index_to_label), predict_cow(probes[2], model, index_to_label)]
print(f"Predicted cow A: {predicted_cow[0]}")
print(f"Predicted cow B: {predicted_cow[1]}")
print(f"Predicted cow C: {predicted_cow[2]}")