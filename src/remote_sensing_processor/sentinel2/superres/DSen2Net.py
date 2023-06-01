import torch
import torch.nn as nn
import torch.nn.functional as F


class DSen2Net(nn.Module):
    def __init__(self, input_shape, num_layers = 32, feature_size = 256):
        super(DSen2Net, self).__init__()
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.conv1 = nn.Conv2d(sum(x[0] for x in input_shape), feature_size, kernel_size=3, padding='same')
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Conv2d(feature_size, feature_size, kernel_size=3, padding='same'))
            torch.nn.init.kaiming_uniform_(self.layers[i*2].weight)
            self.layers.append(nn.Conv2d(feature_size, feature_size, kernel_size=3, padding='same'))
            torch.nn.init.kaiming_uniform_(self.layers[i*2+1].weight)
        self.conv2 = nn.Conv2d(feature_size, input_shape[-1][0], kernel_size=3, padding='same')
        torch.nn.init.kaiming_uniform_(self.conv2.weight)


    def forward(self, inputs):
        if len(self.input_shape) == 3:
            combined = torch.cat((inputs[0], inputs[1], inputs[2]), dim=1)
        else:
            combined = torch.cat((inputs[0], inputs[1]), dim=1)
        x = self.conv1(combined)
        x = F.relu(x)
        for i in range(self.num_layers):
            tmp = self.layers[i*2](x)
            tmp = F.relu(tmp)
            tmp = self.layers[i*2+1](tmp)
            tmp = torch.mul(tmp, 0.1)
            x = torch.add(x, tmp)
        x = self.conv2(x)
        if len(self.input_shape) == 3:
            x = torch.add(x, inputs[2])
        else:
            x = torch.add(x, inputs[1])
        return x


"""from __future__ import division
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Concatenate, Activation, Lambda, Add, Input
import tensorflow.keras.backend as K

K.set_image_data_format('channels_first')


def resBlock(x, channels, kernel_size=[3, 3], scale=0.1):
    tmp = Conv2D(channels, kernel_size, kernel_initializer='he_uniform', padding='same')(x)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(channels, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([x, tmp])


def s2model(input_shape, num_layers=32, feature_size=256):

    input10 = Input(shape=input_shape[0])
    input20 = Input(shape=input_shape[1])
    if len(input_shape) == 3:
        input60 = Input(shape=input_shape[2])
        x = Concatenate(axis=1)([input10, input20, input60])
    else:
        x = Concatenate(axis=1)([input10, input20])

    # Treat the concatenation
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same')(x)

    for i in range(num_layers):
        x = resBlock(x, feature_size)

    # One more convolution, and then we add the output of our first conv layer
    x = Conv2D(input_shape[-1][0], (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    # x = Dropout(0.3)(x)
    if len(input_shape) == 3:
        x = Add()([x, input60])
        model = Model(inputs=[input10, input20, input60], outputs=x)
    else:
        x = Add()([x, input20])
        model = Model(inputs=[input10, input20], outputs=x)
    return model"""

