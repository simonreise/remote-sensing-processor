import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


def unet(input_shape, input_dims, num_classes):
    inputs = layers.Input(shape = (input_shape, input_shape, input_dims))

    conv1 = layers.Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(inputs)
    norm1 = layers.BatchNormalization()(conv1)
    relu1 = layers.Activation('relu')(norm1)
    conv1 = layers.Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(relu1)
    norm1 = layers.BatchNormalization()(conv1)
    relu1 = layers.Activation('relu')(norm1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(relu1)
    drop1 = layers.Dropout(0.1)(pool1)
    conv2 = layers.Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(drop1)
    norm2 = layers.BatchNormalization()(conv2)
    relu2 = layers.Activation('relu')(norm2)
    conv2 = layers.Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(relu2)
    norm2 = layers.BatchNormalization()(conv2)
    relu2 = layers.Activation('relu')(norm2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(relu2)
    drop2 = layers.Dropout(0.1)(pool2)
    conv3 = layers.Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(drop2)
    norm3 = layers.BatchNormalization()(conv3)
    relu3 = layers.Activation('relu')(norm3)
    conv3 = layers.Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(relu3)
    norm3 = layers.BatchNormalization()(conv3)
    relu3 = layers.Activation('relu')(norm3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(relu3)
    drop3 = layers.Dropout(0.1)(pool3)
    conv4 = layers.Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(drop3)
    norm4 = layers.BatchNormalization()(conv4)
    relu4 = layers.Activation('relu')(norm4)
    conv4 = layers.Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(relu4)
    norm4 = layers.BatchNormalization()(conv4)
    relu4 = layers.Activation('relu')(norm4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(relu4)
    drop4 = layers.Dropout(0.1)(pool4)

    conv5 = layers.Conv2D(1024, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(drop4)
    norm5 = layers.BatchNormalization()(conv5)
    relu5 = layers.Activation('relu')(norm5)
    conv5 = layers.Conv2D(1024, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(relu5)
    norm5 = layers.BatchNormalization()(conv5)
    relu5 = layers.Activation('relu')(norm5)
    up6 = layers.Conv2DTranspose(512, (2, 2), strides = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(relu5)
    merge6 = layers.concatenate([relu4,up6], axis = 3)
    drop6 = layers.Dropout(0.1)(merge6)
    conv6 = layers.Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(drop6)
    norm6 = layers.BatchNormalization()(conv6)
    relu6 = layers.Activation('relu')(norm6)
    conv6 = layers.Conv2D(512, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(relu6)
    norm6 = layers.BatchNormalization()(conv6)
    relu6 = layers.Activation('relu')(norm6)

    up7 = layers.Conv2DTranspose(256, (2, 2), strides = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(relu6)
    merge7 = layers.concatenate([relu3,up7], axis = 3)
    drop7 = layers.Dropout(0.1)(merge7)
    conv7 = layers.Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(drop7)
    norm7 = layers.BatchNormalization()(conv7)
    relu7 = layers.Activation('relu')(norm7)
    conv7 = layers.Conv2D(256, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(relu7)
    norm7 = layers.BatchNormalization()(conv7)
    relu7 = layers.Activation('relu')(norm7)
    up8 = layers.Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(relu7)
    merge8 = layers.concatenate([relu2,up8], axis = 3)
    drop8 = layers.Dropout(0.1)(merge8)
    conv8 = layers.Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(drop8)
    norm8 = layers.BatchNormalization()(conv8)
    relu8 = layers.Activation('relu')(norm8)
    conv8 = layers.Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(relu8)
    norm8 = layers.BatchNormalization()(conv8)
    relu8 = layers.Activation('relu')(norm8)
    up9 = layers.Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(relu8)
    merge9 = layers.concatenate([relu1,up9], axis = 3)
    drop9 = layers.Dropout(0.1)(merge9)
    conv9 = layers.Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(drop9)
    norm9 = layers.BatchNormalization()(conv9)
    relu9 = layers.Activation('relu')(norm9)
    conv9 = layers.Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal')(relu9)
    norm9 = layers.BatchNormalization()(conv9)
    relu9 = layers.Activation('relu')(norm9)

    #conv9 = layers.Conv2D(2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv10 = layers.Conv2D(1, 1, activation = 'softmax', padding = 'same')(conv9)

    conv10 = layers.Conv2D(num_classes, (1, 1), activation = 'softmax', padding = 'same')(relu9)

    model = models.Model(inputs = inputs, outputs = conv10)
    
    return model
    
    

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)
    
    
def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)
    
    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)
    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def deeplabv3(input_shape, input_dims, num_classes):
    model_input = layers.Input(shape=(image_size, image_size, input_dims))
    resnet50 = tf.keras.applications.ResNet50(
        weights=None, include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)
    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
    
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), activation = 'softmax', padding="same")(x)
    return models.Model(inputs=model_input, outputs=model_output)


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(mlp_dim, activation=tfa.activations.gelu),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1


class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.rescale = Rescaling(1.0 / 255)
        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, num_patches + 1, d_model)
        )
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))
        self.patch_proj = layers.Dense(d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        self.mlp_head = tf.keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(mlp_dim, activation=tfa.activations.gelu),
                layers.Dropout(dropout),
                layers.Dense(num_classes),
            ]
        )

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        x = self.rescale(x)
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)

        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        # First (class token) is used for classification
        x = self.mlp_head(x[:, 0])
        return x
        
        
def vision_transformer(input_shape, input_dims, num_classes):
    model = VisionTransformer(
        image_size=input_shape,
        patch_size=4,
        num_layers=4,
        num_classes=num_classes,
        d_model=64,
        num_heads=4,
        mlp_dim=128,
        channels=input_dims,
        dropout=0.1,
    )
    return model