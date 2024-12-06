from data.data_preprocessing import get_image_generators
from data.data_visualization import display_images
from model.model_definition import build_model
from model.model_training import train_model
from model.model_evaluation import evaluate_model, plot_training_history, plot_confusion_matrix

# Dataset directories and parameters
train_dir = '/kaggle/input/microscopic-fungi-images/train'
valid_dir = '/kaggle/input/microscopic-fungi-images/valid'
test_dir = '/kaggle/input/microscopic-fungi-images/test'
img_width, img_height = 64, 64

# Data preprocessing
train_generator, valid_generator, test_generator = get_image_generators(train_dir, valid_dir, test_dir, img_width, img_height)

# Visualize dataset
display_images(train_dir)

# Build and train model
model = build_model(img_width, img_height, num_classes=5)
history = train_model(model, train_generator, valid_generator)

# Evaluate model
evaluate_model(model, test_generator)
plot_training_history(history)
