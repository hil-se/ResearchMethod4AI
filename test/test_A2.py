import unittest
from unittest.mock import Mock
from src.vgg_pre import VGG_Pre
import numpy as np

class TestA2(unittest.TestCase):

    def setUp(self):
        self.vgg = VGG_Pre(pretrained="test")

    def test_active_query_3(self):
        self.vgg.decision_function = Mock(return_value=np.array([0.81,0.83,0.56,0.31,0.12,0.67,0.1,0.2,1,0.12,0.9]))
        self.vgg.predict = Mock(return_value=np.array([1,1,1,0,0,1,0,0,1,0,1]))
        np.testing.assert_array_equal(self.vgg.active_query(None, 3), np.array([2,5,3]))

    def test_active_query_5(self):
        self.vgg.decision_function = Mock(return_value=np.array([0.81,0.83,0.56,0.31,0.12]))
        self.vgg.predict = Mock(return_value=np.array([1,1,1,0,0]))
        np.testing.assert_array_equal(self.vgg.active_query(None, 10), np.array([2,3,0,1,4]))