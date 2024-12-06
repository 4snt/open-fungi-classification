import os
import tensorflow as tf

def get_image_generators(train_dir, valid_dir, test_dir, img_width, img_height):
    train_datagen = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(img_width, img_height),
        batch_size=32,
        label_mode='categorical'
    )
    valid_datagen = tf.keras.preprocessing.image_dataset_from_directory(
        valid_dir,
        image_size=(img_width, img_height),
        batch_size=32,
        label_mode='categorical'
    )
    test_datagen = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(img_width, img_height),
        batch_size=32,
        label_mode='categorical'
    )

    return train_datagen, valid_datagen, test_datagen
