import unittest
from src.preprocessor import Reweighing
import numpy as np

class TestA4(unittest.TestCase):

    def test_Reweighing(self):
        X = {1: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 2: [1,1,0,0,1,0,1,0,1,1,1]}
        y = np.array([1,1,1,0,0,1,0,0,1,0,1])
        A = [1, 2]
        np.testing.assert_allclose(Reweighing(X,y,A), np.array([0.594595, 0.594595, 1.189189, 0.990991, 0.825826, 1.189189,  0.825826, 0.990991, 1.486486, 0.825826, 1.486486]), 0.001)