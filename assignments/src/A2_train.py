from data_reader import load_scut
from vgg_pre import VGG_Pre
from metrics import Metrics
import numpy as np


# Load the new training data
train, protected = load_scut("../data/train.csv")
X = np.array([pixel for pixel in train['pixels']])/255.0
y = np.array(train["Rating"])
# Load the previous trained model
model = VGG_Pre("./checkpoint/attractiveness.keras")
# Fine-tune the model on the new training data
model.fit(X, y, epochs=10)

# Load test data
test, protected = load_scut("../data/test.csv")
X_test = np.array([pixel for pixel in test['pixels']])/255.0
y_test = np.array(test["Rating"])
y_pred = model.predict(X_test)

# Evaluate the prediction performance on the test data with accuracy
m = Metrics(y_test, y_pred)
print("Accuracy: %.2f" %m.acc())

# Run A2_query.py file again to query more oracles


