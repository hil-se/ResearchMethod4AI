import unittest
from src.vgg_pre import VGG_Pre
import numpy as np
import tensorflow as tf
import pickle

class TestA5(unittest.TestCase):

    def setUp(self):
        self.vgg = VGG_Pre(pretrained="ImageNet")
        self.vgg.model =  tf.keras.Model(inputs=self.vgg.model.input, outputs=self.vgg.model.layers[-3].output)

    def test_Reweighing(self):
        X = tf.cast(np.ones((224, 224, 3))/2, tf.float32)
        input = tf.Variable([X])
        with open("test/A5_result.pkl", "rb") as f:
            result = pickle.load(f)
        np.testing.assert_allclose(self.vgg.output_grad(input), result, atol=0.01)
