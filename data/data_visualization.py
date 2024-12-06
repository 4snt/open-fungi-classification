import os
import cv2
import matplotlib.pyplot as plt

def display_images(dataset_dir):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    axs = axs.ravel()
    for i, class_folder in enumerate(os.listdir(dataset_dir)):
        img_path = os.path.join(dataset_dir, class_folder, os.listdir(os.path.join(dataset_dir, class_folder))[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img)
        axs[i].set_title(f"Class: {class_folder}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()
