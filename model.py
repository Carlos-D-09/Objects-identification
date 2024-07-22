import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from PIL import Image
import numpy as np

"""
    Convert image PNG with transparency into RGBA
    Parameters:
    directory (str): Directory name containing the images. 
"""
def convert_to_rgba(directory):
    
    # Recursively traverse each subdirectory in the given directory
    for root, _, files in os.walk(directory): 
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Try to open the image with the constructed file path
                img = Image.open(file_path)
                
                # Case if the image is in palette mode ('P') and has transparency
                if img.mode == 'P' and 'transparency' in img.info:
                    img = img.convert('RGBA')
                    img.save(file_path)
            except Exception as e:
                print(f'Error processing {file_path}: {e}')

"""
    Test a valid Keras model using the following metrics: precision, recall, and accuracy.
    
    Parameters:
    model (tf.keras.Model): A valid Keras model.
    test (ImageDataGenerator): A valid dataset consistent with the training dataset.
"""
def test_model(model, test):
    print('Testing model')
    
    #Evaluation metrics
    pre = Precision()
    re = Recall()
    acc = CategoricalAccuracy()

    for batch in test:
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    
    print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy:{acc.result().numpy()}')
        

def main():
    #GPU settings
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        print("GPU wasn't find. TensorFlow will use the CPU.")
    else:
        tf.keras.backend.clear_session()
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU detected and configured to be used")

    # Convertir las imágenes a formato RGBA si es necesario
    convert_to_rgba("test")
    convert_to_rgab("train")

    #Data generator
    trainer = ImageDataGenerator(validation_split=0.2)
    test_data_gen = ImageDataGenerator()

    #Split and load train data from train dataset
    train_data = trainer.flow_from_directory(
        "train",
        target_size=(256, 256),
        batch_size=10,
        class_mode='categorical',
        subset='training'
    )

    #Split and load validation data from train dataset
    val_data = trainer.flow_from_directory(
        "train",
        target_size=(256, 256),
        batch_size=10,
        class_mode='categorical',
        subset='validation'
    )

    #Load test data set
    test_data = test_data_gen.flow_from_directory(
        "test",
        target_size=(256, 256),
        batch_size=10,
        class_mode='categorical',
        shuffle=False
    )

    #Load and setting InceptionV3
    inception = applications.InceptionV3(include_top=False, input_shape=(256,256,3))

    #Model to predict according to the train data
    predictor = Sequential([
        Flatten(),
        Dense(128, activation="relu"),
        Dense(train_data.num_classes, activation="softmax")
    ])

    #Create model
    model = Sequential([inception, predictor])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    #Train model
    model.fit(train_data, epochs=50, validation_data=val_data)
    print('Model trained successfully')

    #Save model
    model.save('objects_model.keras')
    print('Model saved successfully')

    #Load model
    model = load_model('objects_model.keras')
    model.summary()
    print('Model loaded successfully')
    
    #Test model
    test_model(model, test_data)

main()