from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    # X: independent variables (2-d pd.DataFrame)
    # Y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: a list/array of the names of the sensitive attributes with binary values
    # Return: sample_weight, an array of float weight for every data point
    #         sample_weight(a,y) = P(y)*P(a)/P(a,y)
    # Write your code below:


    # Rescale the sum of sample weights to len(y) before returning it
    sample_weight = sample_weight * len(Y) / sum(sample_weight)
    return sample_weight


