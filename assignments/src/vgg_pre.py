import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Change the number 0 to your corresponding GPU ID in the Google Sheet

# Change the following path if you are not running on CS Clusters
weight_path = "/local/datasets/idai720/checkpoint/vgg_face_weights.h5"

class VGG_Pre:
    def __init__(self, pretrained = "ImageNet"):
        if pretrained == "ImageNet":
            # Use the same structure as the pre-trained VGG-16 model
            start_size = 64
            input_shape = (224, 224, 3)
            base_model = tf.keras.models.Sequential()
            base_model.add(
                tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                       input_shape=input_shape))
            base_model.add(
                tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                       input_shape=input_shape))
            base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            base_model.add(
                tf.keras.layers.Conv2D(start_size*2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
            base_model.add(
                tf.keras.layers.Conv2D(start_size * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       activation='relu'))
            base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            base_model.add(
                tf.keras.layers.Conv2D(start_size*4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
            base_model.add(
                tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       activation='relu'))
            base_model.add(
                tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       activation='relu'))
            base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            base_model.add(
                tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       activation='relu'))
            base_model.add(
                tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       activation='relu'))
            base_model.add(
                tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       activation='relu'))
            base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            base_model.add(
                tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       activation='relu'))
            base_model.add(
                tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       activation='relu'))
            base_model.add(
                tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       activation='relu'))
            base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

            base_model.add(
                tf.keras.layers.Conv2D(4096, kernel_size=(7, 7), strides=(1, 1), padding='valid',
                                       activation='relu'))
            base_model.add(tf.keras.layers.Dropout(0.5))
            base_model.add(
                tf.keras.layers.Conv2D(4096, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                       activation='relu'))
            base_model.add(tf.keras.layers.Dropout(0.5))
            base_model.add(
                tf.keras.layers.Conv2D(2622, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                       activation='relu'))

            base_model.add(tf.keras.layers.Flatten())
            base_model.add(tf.keras.layers.Activation('softmax'))
            # Load the pre-trained weights
            # you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
            # related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
            base_model.load_weights(weight_path)

            # Discard the output layers of the pre-trained model
            base_model_output = tf.keras.layers.Flatten()(base_model.layers[-4].output)

            # Replace with a dense layer and a sigmoid output layer
            base_model_output = tf.keras.layers.Dense(256, activation='relu')(base_model_output)
            base_model_output = tf.keras.layers.Dense(1, activation='sigmoid')(base_model_output)

            # Compile the model so that it optimizes for binary cross-entropy loss with stochastic gradient descent
            self.model = tf.keras.Model(inputs=base_model.input, outputs=base_model_output)
            self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'], optimizer='SGD')
        elif pretrained=="test":
            pass
        else:
            self.load_model(pretrained)

    def fit(self, X, y, sample_weight=None, epochs = 50, batch_size = 10):
        # Fit the model on training data (X, y) by using 20% of the data as validation data
        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, mode='auto',
                                                         min_lr=5e-5)
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint/attractiveness.keras'
                                                          , monitor="val_loss", verbose=1
                                                          , save_best_only=True, mode='auto'
                                                          )
        # Split data into training and validation to avoid overfitting
        n = len(y)
        # Randomly select 20% as validation
        val_ind = np.random.choice(range(n), int(0.2*n), replace=False)
        train_ind = np.array(list(set(range(n))-set(list(val_ind))))
        # Sample weight for training and validation
        if sample_weight is None:
            train_weight = None
            val_weight = None
        else:
            train_weight = sample_weight[train_ind]
            val_weight = sample_weight[val_ind]
        # Fit model
        self.model.fit(X[train_ind], y[train_ind], sample_weight=train_weight, callbacks=[lr_reduce, checkpointer],
                       validation_data=(X[val_ind], y[val_ind], val_weight), batch_size=batch_size, epochs=epochs, verbose=1)
        # Load the model weights with lowest val_loss
        self.load_model('checkpoint/attractiveness.keras')

    def predict(self, X):
        # Make predictions (binary classes) on input data X
        pred = self.decision_function(X)
        pred = (pred.flatten()>0.5).astype(int).astype(float)
        return pred

    def decision_function(self, X):
        # The model's decision function (prediction probabilities from [0.0, 1.0]) on input data X
        pred = self.model.predict(X)
        return pred

    def load_model(self, checkpoint_filepath):
        # Load a pretrained model from checkpoint_filepath
        self.model = tf.keras.models.load_model(checkpoint_filepath)

    # Below is for A2
    def active_query(self, X, k=10):
        # X: input to the model of the unlabeled data
        # k: number of data points selected to query for oracles
        # Return the indices of top k most uncertain predictions
        # Write your code below:

        return inds[:k]

    # Below is for A5
    def output_grad(self, inputs):
        # inputs: tf.Variable([one input data point])
        # Return grad: gradients from every input node to the output of self.decision_function(X) or self.model(X) (numpy.array),
        # Write your code below:

        return grad.numpy()[0]



