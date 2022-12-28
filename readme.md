This script is used to evaluate the performance of a trained model on the MNIST dataset. The model is first loaded from a file my_model.h5, and the MNIST dataset is loaded and normalized. The model is then used to make predictions on the test data, and the accuracy is calculated using the Accuracy metric from TensorFlow.

Finally, the script displays 10 random test samples and their predictions. For each sample, a prediction is made using the model, and the test sample and prediction are plotted using Matplotlib.

To run this script, you will need to have the following packages installed:

    TensorFlow
    Numpy
    Matplotlib



To use this script, simply run the following command:

python evaluate_model.py