from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
import os

# ============ DATA =================
# Update these paths to match your dataset!
train_dir = 'C:/Users/Parth/Desktop/Geeti/Tumour/train'
val_dir   = 'C:/Users/Parth/Desktop/Geeti/Tumour/valid'

batch_size = 32
img_size = (224, 224)
num_classes = 4  # brain tumor classes

# Data generators for grayscale images (custom CNN)
train_gen_gray = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=img_size, color_mode='grayscale',
    batch_size=batch_size, class_mode='categorical'
)
val_gen_gray = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=img_size, color_mode='grayscale',
    batch_size=batch_size, class_mode='categorical'
)

# Data generators for RGB images (ResNet50)
train_gen_rgb = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=img_size, color_mode='rgb',
    batch_size=batch_size, class_mode='categorical'
)
val_gen_rgb = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=img_size, color_mode='rgb',
    batch_size=batch_size, class_mode='categorical'
)

# ============ CUSTOM CNN MODEL =================
custom_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
custom_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

print("Training CUSTOM CNN (grayscale input)...")
custom_model.fit(
    train_gen_gray,
    epochs=15,
    validation_data=val_gen_gray
)
custom_model.save('custom_cnn_model.h5')
print("Custom CNN model saved as custom_cnn_model.h5")

# ============== PRETRAINED MODEL eg. RESNET50 ==============
# Note: ResNet50 expects RGB input
print("Building and training ResNet50 (RGB input, transfer learning)...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False  # freeze pretrained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)
resnet_model = Model(inputs=base_model.input, outputs=output)

resnet_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

resnet_model.fit(
    train_gen_rgb,
    epochs=10,
    validation_data=val_gen_rgb
)
resnet_model.save('resnet50_model.h5')
print("ResNet50 model saved as resnet50_model.h5")