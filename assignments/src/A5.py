from data_reader import load_scut
from vgg_pre import VGG_Pre
import numpy as np
import tensorflow as tf
from pdb import set_trace

# Load the previous trained model
model = VGG_Pre("./checkpoint/attractiveness.keras")

# Load test data
test, protected = load_scut("../data/test.csv")
X = np.array([pixel for pixel in test['pixels']])/255.0
y = np.array(test["Rating"])

for i in range(len(y)):
    # Explain the model around the ith test data point
    inputs = tf.Variable([X[i]])
    file_name = test["Filename"][i].split('.')[0]
    grad = model.output_grad(inputs)
    pil_img = tf.keras.utils.array_to_img(grad, scale=True)
    pil_img.save("../explain/" + str(file_name) + ".jpg")
    original_img = tf.keras.utils.array_to_img(X[i], scale=True)
    original_img.save("../explain/" + str(file_name) + "_original.jpg")

