import os
from keras.preprocessing.image import ImageDataGenerator

def get_image_generators(train_dir, valid_dir, test_dir, img_width, img_height):
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
    
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(img_width, img_height), batch_size=32, class_mode='categorical', shuffle=True
    )
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir, target_size=(img_width, img_height), batch_size=32, class_mode='categorical', shuffle=True
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(img_width, img_height), batch_size=32, class_mode='categorical'
    )
    return train_generator, valid_generator, test_generator
