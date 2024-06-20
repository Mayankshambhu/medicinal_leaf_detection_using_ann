import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Define the training directory
train_dir = 'C:/Users/mayan/PycharmProjects/pythonProject/Indian Medicinal Leaves Image Datasets/Medicinal Leaf dataset'

# Define the image data generator
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

class_labels = list(train_generator.class_indices.keys())
num_classes = len(class_labels)

# Define the CNN model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=50
)

# Save the trained model to a file
model.save('medicinal_plant_model.h5')

# Define a function to preprocess the input image
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = img / 255.0
    return img

# Define a function to identify medicinal plants
def identify_plant(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    return class_labels[class_id]

# Test the model
image_path = 'C:/Users/mayan/PycharmProjects/pythonProject/pngtree-coffee-leaves-png-file-png-image_11534129.png'
class_label = identify_plant(image_path)
print(f'Identified plant: {class_label}')