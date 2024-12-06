from data.data_preprocessing import get_image_generators
from data.data_visualization import display_images
from model.model_definition import build_model
from model.model_training import train_model
from model.model_evaluation import evaluate_model, plot_training_history, plot_confusion_matrix

# Diretórios e parâmetros do dataset
train_dir = './data/microscopic-fungi-images/train'
valid_dir = './data/microscopic-fungi-images/valid'
test_dir = './data/microscopic-fungi-images/test'
img_width, img_height = 64, 64

# Pré-processamento dos dados
train_generator, valid_generator, test_generator = get_image_generators(
    train_dir, valid_dir, test_dir, img_width, img_height
)

# Visualização do dataset
display_images(train_dir)

# Construir e treinar o modelo
model = build_model(img_width, img_height, num_classes=5)
history = train_model(model, train_generator, valid_generator)

# Avaliar o modelo
evaluate_model(model, test_generator)
plot_training_history(history)
plot_confusion_matrix(model, test_generator)
