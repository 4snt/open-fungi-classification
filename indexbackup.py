#Import Necessary Libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from keras.regularizers import l2
from keras import backend as K
import cv2
import tensorflow as tf

#Define Dataset Directories

train_dir = '/kaggle/input/microscopic-fungi-images/train'
valid_dir = '/kaggle/input/microscopic-fungi-images/valid'
test_dir = '/kaggle/input/microscopic-fungi-images/test'

#Explore Dataset Structure
# Print the number of images in each set and each class
for dataset in [train_dir, valid_dir, test_dir]:
    print(f"Dataset: {dataset.split('/')[-1]}")
    for class_folder in os.listdir(dataset):
        print(f"Class: {class_folder}, Number of images: {len(os.listdir(os.path.join(dataset, class_folder)))}")
    print("\n")
    
#Visualize Some Images
# Function to display one image from each class
def display_images(dataset_dir):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))  # Changed to create only 5 subplots
    axs = axs.ravel()
    for i, class_folder in enumerate(os.listdir(dataset_dir)):
        img_path = os.path.join(dataset_dir, class_folder, os.listdir(os.path.join(dataset_dir, class_folder))[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img)
        axs[i].set_title(f"Class: {class_folder}")
        axs[i].axis('off')
    plt.tight_layout()

# Display one image from each class in the training set
display_images(train_dir)

# Grad-CAM Visualization Function

def grad_cam(model, image, cls, layer_name):
    """Generate a heatmap via Grad-CAM for a specific class."""
    with tf.GradientTape() as tape:
        # Watch the convolutional output and get the model's prediction
        tape.watch(model.input)
        y_c = model.output[0, cls]
        conv_output = model.get_layer(layer_name).output

    # Compute the gradient of the class output value with respect to the feature map
    grads = tape.gradient(y_c, conv_output)

    # Pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # Multiply the pooled gradients with the convolutional output to get the heatmap
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)

    # Process the heatmap to make it visually comprehensible
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap.numpy()

# Data Preprocessing
#Define Image Data Generators

# Define Image Dimensions
img_width, img_height = 64, 64

# Data Augmentation with varied parameters
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect' 
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
#Load Images and Apply Data Generators
# Load images with stratified sampling 
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

#Model Building using Transfer Learning with VGG16
# Load VGG16 Model + Higher Level Layers for Transfer Learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation=LeakyReLU(alpha=0.1))(x)  # Using LeakyReLU
x = Dropout(0.5)(x)
x = Dense(256, activation=LeakyReLU(alpha=0.1))(x)  # Additional Dense layer
x = Dropout(0.5)(x)
predictions = Dense(5, activation='softmax', kernel_regularizer=l2(0.01))(x)  # L2 regularization

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the Model with Adam optimizer with learning rate decay
optimizer = Adam(lr=0.0001, decay=1e-6)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Fine-tuning: Unfreeze some layers of the base model
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Re-compile the Model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

#Model Evaluation
#Evaluate the Model on Test Set

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=np.ceil(test_generator.samples / test_generator.batch_size))
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

#Predict Classes for Test Set

# Predict classes for test set
test_generator.reset()
pred = model.predict(test_generator, steps=np.ceil(test_generator.samples / test_generator.batch_size), verbose=1)
predicted_classes = np.argmax(pred, axis=1)


#Print Classification Report

# Print classification report
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

##Plot Confusion Matrix

# Plot confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Interpretation and Communication of Results

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()