import unittest
from src.metrics import Metrics
import numpy as np

class TestA3(unittest.TestCase):

    def setUp(self):
        y = np.array([1,1,1,0,0,1,0,0,1,0,1])
        y_pred = np.array([1,0,1,0,1,1,0,1,0,0,0])
        self.m = Metrics(y, y_pred)

    def test_eod(self):
        s = [1,1,1,1,0,0,0,0,0,0,0]
        self.assertAlmostEqual(self.m.eod(s), 0.333, 3)

    def test_aod(self):
        s = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        self.assertAlmostEqual(self.m.aod(s), -0.083, 3)

    def test_spd(self):
        s = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        self.assertAlmostEqual(self.m.spd(s), 0.071, 3)