# predict_and_save.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import csv

# --------- SETTINGS ----------
base_dir = r'C:/Users/Parth/Desktop/Geeti/Tumour'  # Update path if needed!
splits = ['train', 'valid', 'test']
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
img_size = (224, 224)

def predict_and_save(model, out_csv, color_mode):
    results = []
    for split in splits:
        split_path = os.path.join(base_dir, split)
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.jpg','.jpeg','.png')):
                        img_path = os.path.join(class_path, filename)
                        # ---- Set color_mode for each model ----
                        img = image.load_img(img_path, target_size=img_size, color_mode=color_mode)
                        img_array = image.img_to_array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        prediction = model.predict(img_array)
                        predicted_class = int(np.argmax(prediction))
                        results.append([
                            split, 
                            class_name, 
                            filename, 
                            class_names[predicted_class]
                        ])
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'true_class', 'filename', 'predicted_class'])
        writer.writerows(results)
    print(f"Predictions saved to {out_csv}")

# --------- LOAD MODELS ----------
# Update these paths if model files are elsewhere
custom_model = load_model('custom_cnn_model.h5')
resnet_model = load_model('resnet50_model.h5')

# --------- RUN PREDICTIONS ----------
# For custom CNN (expecting grayscale images):
predict_and_save(custom_model, 'predictions_custom.csv', color_mode='grayscale')

# For ResNet (expecting RGB images):
predict_and_save(resnet_model, 'predictions_resnet.csv', color_mode='rgb')