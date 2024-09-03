from data_reader import load_scut
from vgg_pre import VGG_Pre
from metrics import Metrics
import numpy as np

# Load training data
train, protected = load_scut("../data/train.csv")
X = np.array([pixel for pixel in train['pixels']])/255.0
y = np.array(train["Rating"])
model = VGG_Pre()
# Fine-tune the model on the training data for the rating task
model.fit(X, y, epochs=50)

# Load test data
test, protected = load_scut("../data/test.csv")
X_test = np.array([pixel for pixel in test['pixels']])/255.0
y_test = np.array(test["Rating"])
y_pred = model.predict(X_test)

# Evaluate the prediction performance on the test data with accuracy
m = Metrics(y_test, y_pred)
print("Accuracy: %.2f" %m.acc())