from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from PIL import Image
import tensorflow as tf
import os
import tkinter as tk
from tkinter import filedialog

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the input data
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalize pixel values to range between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 128
epochs = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Save the model
model.save("model.h5")
print("Model saved.")

# Load the model
loaded_model = tf.keras.models.load_model("model.h5")
print("Model loaded.")

# Evaluate the model on the test dataset
test_loss, test_accuracy = loaded_model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

def predict_digit(filename):
    # Open the image file
    img = Image.open(filename).convert("L")

    # Resize the image to 28x28 pixels
    img = img.resize((28,28))

    # Convert the image to numpy array and reshape it to expected model input shape
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)

    # Normalize pixel values to range between 0 and 1
    img = img.astype('float32') / 255.0

    # Use the model to predict the digit
    digit = loaded_model.predict(img)
    digit = digit.argmax() # Get the index of the maximum value

    return digit

# Create a new Tkinter window
window = tk.Tk()

# Create a new button on the window that will open the file dialog
open_button = tk.Button(window, text="Open image", command=lambda: open_image())
open_button.pack()

def open_image():
    # Open the file dialog and get the path of the selected file
    file_path = filedialog.askopenfilename()

    # Make sure a file was selected
    if file_path:
        # Use the model to predict the digit
        digit = predict_digit(file_path)
        print("Predicted digit:", digit)

# Run the Tkinter event loop
window.mainloop()
