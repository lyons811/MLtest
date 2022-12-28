import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model from a file
model = tf.keras.models.load_model('my_model.h5')

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the input data
x_test = x_test / 255.0

# Make predictions on the test data
predictions = model.predict(x_test)

# Convert the predictions to integer labels
predicted_labels = np.argmax(predictions, axis=-1)

# Calculate the model's accuracy
accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(y_test, predicted_labels)
print('Accuracy:', accuracy.result().numpy())

# Display 10 random test samples and their predictions
# Display 10 random test samples and their predictions
for i in range(10):
    # Select a random test sample
    sample_index = random.randint(0, len(x_test))
    sample_image = x_test[sample_index]
    sample_label = y_test[sample_index]

    # Make a prediction on the test sample
    prediction = model.predict(sample_image.reshape(1, 28, 28))
    predicted_label = np.argmax(prediction)

    # Plot the test sample and the prediction
    plt.figure()
    plt.imshow(sample_image, cmap='gray')
    plt.title('Prediction: {}'.format(predicted_label))
    plt.show()
    plt.close()
