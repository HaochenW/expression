import numpy as np
import pandas as pd
import keras
import cv2
from vgg import VGG

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=3,
                 n_classes=7, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        model_image_size = self.dim
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = cv2.imread('../all/origin/' + ID)
            try:
                resized_image = cv2.resize(img, model_image_size, interpolation=cv2.INTER_CUBIC)
            except:
                print(ID)
            resized_image = resized_image.astype(np.float32)
            resized_image /= 255.

            X[i,] = resized_image
            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# Parameters
params = {'dim': (160,160),
          'batch_size': 32,
          'n_classes': 7,
          'n_channels': 3,
          'shuffle': True}


# Datasets
def get_dataset_info():
    train_info = pd.read_table('train_info.txt', sep= ' ', header = None)
    val_info = pd.read_table('val_info.txt', sep= ' ', header = None)

    partition = {'train':train_info[0].tolist(), 'validation': val_info[0].tolist()}   #建立所有数据的名称集
    info = pd.concat([train_info,val_info])[[0,1]]
    labels = info.set_index(0).T.to_dict('list')
    for keys in labels:
        labels[keys] = labels[keys][0]
    return partition,labels

partition,labels = get_dataset_info()

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
custom_model = VGG()
for layer in custom_model.layers[:7]:
    layer.trainable = False
custom_model.compile(loss='categorical_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])


# Train model on dataset
custom_model.fit_generator(generator=training_generator, epochs=100,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)

