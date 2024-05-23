from data_reader import load_scut
from vgg_pre import VGG_Pre
from metrics import Metrics
import numpy as np

# Load the previous trained model
model = VGG_Pre("./checkpoint/attractiveness.keras")

# Load test data
test, protected = load_scut("../data/test.csv")
X_test = np.array([pixel for pixel in test['pixels']])/255.0
y_test = np.array(test["Rating"])
y_pred = model.predict(X_test)

# Evaluate the prediction performance on the test data with accuracy
m = Metrics(y_test, y_pred)
print("Accuracy: %.2f" %m.acc())

# Test Group Fairness
m = Metrics(y_test, y_pred)
for A in protected:
    print("EOD on %s: %.2f" % (A, m.eod(test[A])))
    print("AOD on %s: %.2f" % (A, m.aod(test[A])))
    print("SPD on %s: %.2f" % (A, m.spd(test[A])))
