from flask import Flask, request, render_template, url_for
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# Create a directory to store uploaded images
UPLOAD_DIR = 'uploads'
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load the trained model
model = tf.keras.models.load_model('medicinal_plant_model.h5')

# Load the class indices from the training generator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    'C:/Users/mayan/PycharmProjects/pythonProject/Indian Medicinal Leaves Image Datasets/Medicinal Leaf dataset',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

class_labels = list(train_generator.class_indices.keys())

# Load the plant information from an Excel file
plant_info_df = pd.read_excel('C:/Users/mayan/Downloads/medicinal_plants.xlsx')

# Define a function to identify the plant from an image
def identify_plant_from_image(image_path):
    try:
        # Load the image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (256, 256))
        img = img / 255.0

        # Add a batch dimension
        img = tf.expand_dims(img, 0)

        # Make predictions
        predictions = model.predict(img)

        # Get the class label with the highest probability
        class_label_index = np.argmax(predictions[0])

        # Get the corresponding plant name
        plant_name = class_labels[class_label_index]

        return plant_name
    except Exception as e:
        return str(e)

# Define the route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for image upload and identification
@app.route('/identify', methods=['POST'])
def identify():
    image_file = request.files['image']
    image_path = os.path.join(UPLOAD_DIR, image_file.filename)
    image_file.save(image_path)
    plant_name = identify_plant_from_image(image_path)

    # Retrieve the plant information from the Excel file
    plant_data = plant_info_df.loc[plant_info_df['Plant Name'] == plant_name]
    if not plant_data.empty:
        english_name = plant_data['English Name'].values[0]
        botanical_name = plant_data['Botanical Name'].values[0]
        uses = plant_data['Uses'].values[0]
    else:
        english_name = None
        botanical_name = None
        uses = None

    return render_template('result.html', plant_name=plant_name, english_name=english_name,
                           botanical_name=botanical_name, uses=uses, image_filename=image_file.filename)

if __name__ == '__main__':
    app.run(debug=True)