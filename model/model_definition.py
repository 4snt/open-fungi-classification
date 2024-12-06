from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam

def build_model(img_width, img_height, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
