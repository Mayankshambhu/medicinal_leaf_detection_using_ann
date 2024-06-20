Overview
The Medical Leaf Detection System is a web application designed to identify medicinal plants from images of their leaves. It leverages a Convolutional Neural Network (CNN) for accurate plant identification and provides detailed information about the identified plant, including its English name, botanical name, and uses. The project is built using TensorFlow, Keras, and Flask.

Features
- **CNN Model for Plant Identification**: A trained CNN model to classify medicinal plants from images.
- **Web Interface**: A user-friendly web application built with Flask for uploading images and displaying results.
- **Comprehensive Plant Information**: Detailed information about each identified plant, including its English name, botanical name, and uses.

Requirements
- Python 3.7 or higher
- TensorFlow 2.x
- Flask
- NumPy
- Pandas
- OpenCV (for image processing)
- Anaconda or any other virtual environment manager (recommended)

Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mayankshambhu/medicinal_leaf_detection_using_cnn.git
   cd medicinal_leaf_detection_using_cnn
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the plant dataset and place it in the appropriate directory (update `train_dir` in `file1.py`):
   ```bash
   # Example path
   train_dir = 'path_to_your_dataset/Medicinal Leaf dataset'
   ```

5. Place the trained model file (`medicinal_plant_model.h5`) in the project directory.

6. Ensure the plant information Excel file is placed correctly and update the path in `app.py`:
   ```bash
   plant_info_df = pd.read_excel('path_to_your_excel_file/medicinal_plants.xlsx')
   ```

Usage
1. **Training the Model**:
   If you need to train the model from scratch, run `file1.py`:
   ```bash
   python file1.py
   ```

2. **Running the Web Application**:
   Start the Flask application by running `app.py`:
   ```bash
   python app.py
   ```

3. **Accessing the Application**:
   Open your web browser and go to `http://127.0.0.1:5000/` to access the application. You can upload images of leaves to identify the medicinal plant and view detailed information.

Project Structure
```
medical-leaf-detection-system/
│
├── templates/
│   ├── index.html          # Homepage template
│   ├── result.html         # Result page template
│
├── uploads/                # Directory for uploaded images
│
├── app.py                  # Flask application
├── file1.py                # Model training and testing script
├── requirements.txt        # List of dependencies
├── medicinal_plant_model.h5# Trained model file
├── README.md               # Project documentation
```

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License.

Acknowledgments
- The dataset used for training the model.
- The TensorFlow and Keras libraries for providing robust machine learning tools.
- Flask for creating a seamless web application experience.
