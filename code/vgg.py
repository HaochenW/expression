from keras import applications
from keras.layers import *

def VGG(given_input_shape = (160, 160, 3), initial_weight = 'imagenet'):
    # If you want to specify input tensor
    from keras.layers import Input
    vgg_model = applications.VGG16(weights=initial_weight,
                                   include_top=False,
                                   input_shape=given_input_shape)
    vgg_model.summary()
    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    x = layer_dict['block2_pool'].output

    # Stacking a new simple convolutional network on top of it
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(7, activation='softmax')(x)

    # Creating new model. Please note that this is NOT a Sequential() model.
    from keras.models import Model
    custom_model = Model(input=vgg_model.input, output=x)
    for layer in custom_model.layers[:7]:
        layer.trainable = False
    return custom_model
    # Make sure that the pre-trained bottom layers are not trainable

