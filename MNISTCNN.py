import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape dataset to fit the CNN model
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32') / 255.0

# One-hot encoding for labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN model
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(28,28,1)))
cnn.add(MaxPooling2D())
cnn.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
cnn.add(MaxPooling2D())
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(10, activation='softmax'))

# Compile model
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
print(cnn.summary())

# Train the model
history_cnn = cnn.fit(X_train, y_train, epochs=50, verbose=1, validation_data=(X_test, y_test))


# Plot training & validation accuracy
plt.plot(history_cnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("CNN Model Accuracy on MNIST")
plt.show()

# Evaluate the model
score = cnn.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f}')

# Select an image from the test set
index = 0  
test_image = X_test[index]
true_label = np.argmax(y_test[index])

# Reshape the image to add batch dimension 
test_image = test_image.reshape(1, 28, 28, 1)

# Predict the class of the test image
predicted_label = np.argmax(cnn.predict(test_image))

# Plot the image
plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.title(f'True Value: {true_label}, Predicted Value: {predicted_label}')
plt.show()

print(f'True label: {true_label}')
print(f'Predicted label: {predicted_label}')

