import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
from model_utils import TrackMetrics

########## Gaze-Terrain Network Model ###########
class GazeEncoder(tfk.layers.Layer):
    def __init__(self, output_type='softmax',name="gaze_encoder", **kwargs):
        super(GazeEncoder, self).__init__(name=name, **kwargs)

        self.up_pooling_x2 = tf.keras.layers.UpSampling2D(size=2,interpolation='bilinear')

        self.conv_layer_1 = tfk.layers.Conv2D(filters=256,kernel_size=1, strides=(1,1),
                                              padding='same',activation='relu')
        self.bn_1 = tfk.layers.BatchNormalization()

        self.conv_layer_2 = tfk.layers.Conv2D(filters=1, kernel_size=1, strides=(1,1),
                                              padding='same', activation='linear')

        self.output_type = output_type
        if self.output_type =='softmax':
            self.softmax_layer_1=tfk.layers.Softmax(axis=[1,2])

    # @tf.function(input_signature=
    #              [tf.TensorSpec(shape=(None,7,7,512), dtype=tf.float32),
    #               tf.TensorSpec(shape=(), dtype=tf.bool)])
    def call(self,MN_inter_fea,training=True):

        fea_28 = MN_inter_fea[0]
        fea_14 = MN_inter_fea[1]
        fea_14 = self.conv_layer_1(fea_14)
        fea_14 = self.bn_1(fea_14,training=training)
        fea_14 = self.up_pooling_x2(fea_14)
        att_fea=tf.concat([fea_28,fea_14],axis=3)
        encoded_gaze = self.conv_layer_2(att_fea)

        if self.output_type == 'softmax':
            encoded_gaze = self.softmax_layer_1(encoded_gaze)

        return encoded_gaze

class TerrainDecoder(tfk.layers.Layer):
    def __init__(self, class_num=5,output_type='logit',name="terrain_decoder", **kwargs):
        super(TerrainDecoder, self).__init__(name=name, **kwargs)
        self.GAP_layer = tfk.layers.GlobalAveragePooling2D()
        if output_type == 'prop':
            self.dense_layer_1 = tfk.layers.Dense(units=class_num,activation='softmax',
                                                  kernel_regularizer=tfk.regularizers.l2(1e-5),
                                                  bias_regularizer=tfk.regularizers.l2(1e-5))
        else:
            self.dense_layer_1 = tfk.layers.Dense(units=class_num,activation='linear',
                                                  kernel_regularizer=tfk.regularizers.l2(1e-5),
                                                  bias_regularizer=tfk.regularizers.l2(1e-5))
    # @tf.function(input_signature=[tf.TensorSpec(shape=(None,7,7,512), dtype=tf.float32)])
    def call(self, fused_feature,training=True):
        fused_feature = self.GAP_layer(fused_feature)
        predict_terrain = self.dense_layer_1(fused_feature)
        return predict_terrain

class AttentionModel(tfk.Model):
    def __init__(self,
                 pretrained_model,
                 gaze_main_shape=(28,28,256),
                 class_num=5,
                 name="attention_model",
                 output_type='logit',
                 **kwargs):
        super(AttentionModel, self).__init__(name=name, **kwargs)
        self.pretrained_model = pretrained_model
        self.encoder = GazeEncoder()
        self.decoder = TerrainDecoder(output_type=output_type,class_num=class_num)
        self.ds_ratio = tf.constant(4,tf.float32)
        self.pooling = tfk.layers.AveragePooling2D(pool_size=4)
        self.multiply_layer = tfk.layers.Multiply()

        self.kl_weight = tf.Variable(initial_value=0.,trainable=False,dtype=tf.float32)
        self.gaze_vec_len = tf.convert_to_tensor(gaze_main_shape[0]*gaze_main_shape[1],
                                                 dtype=tf.int64)
        self.class_num = class_num
        self.acc_metric = tfk.metrics.CategoricalAccuracy(name='acc')

        self.kl_metric = tfk.metrics.Mean(name='kl_div')
        self.kl_weight_metric = TrackMetrics(name='kl_weight')


    # @tf.function(input_signature=
    #              [[tf.TensorSpec(shape=None, dtype=tf.float32),
    #               tf.TensorSpec(shape=None, dtype=tf.float32)],
    #              tf.TensorSpec(shape=None, dtype=tf.bool)])
    def call(self,packed_input,training=True):
        MN_inter_fea = self.pretrained_model(packed_input[0])
        posterior_gaze = self.encoder(MN_inter_fea[:-1], training=training)
        tobii_fea = MN_inter_fea[-1]
        posterior_gaze_d4 = self.pooling(posterior_gaze)*self.ds_ratio**2
        ## Simple Overlay ##
        scale_factor = tf.math.reduce_max(posterior_gaze_d4, [1, 2],keepdims=True)
        posterior_gaze_d4 /= scale_factor

        attended_fea = self.multiply_layer([tobii_fea, posterior_gaze_d4])
        prediction = self.decoder(tf.concat([attended_fea,tobii_fea],-1),training=training)
        prior_gaze = packed_input[1]
        posterior_gaze = tf.reshape(posterior_gaze,tf.shape(prior_gaze))
        vec_posterior_gaze = tf.reshape(posterior_gaze,[-1,self.gaze_vec_len])
        vec_prior_gaze = tf.reshape(prior_gaze, [-1, self.gaze_vec_len])
        KLD = tfk.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        kl_loss = tf.math.reduce_mean(KLD(vec_posterior_gaze,vec_prior_gaze))
        self.add_loss(self.kl_weight*kl_loss)
        self.add_metric(self.kl_metric(kl_loss))
        self.add_metric(self.kl_weight_metric(self.kl_weight, self.kl_weight))

        return prediction


    @tf.function
    def train_step(self, data):
        packed_input, one_hot_label = data
        with tf.GradientTape() as tape:
            prediction = self(packed_input,training=True)
            loss = self.compiled_loss(one_hot_label,prediction, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.acc_metric.update_state(one_hot_label, prediction)
        self.compiled_metrics.update_state(one_hot_label, prediction)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        packed_input, one_hot_label = data
        prediction  = self(packed_input,training=False)
        loss = self.compiled_loss(one_hot_label,prediction, regularization_losses=self.losses)
        self.acc_metric.update_state(one_hot_label, prediction)
        self.compiled_metrics.update_state(one_hot_label, prediction)
        return {m.name: m.result() for m in self.metrics}


########## PointNet Model ###########
class PointNetConvLayer(tfk.layers.Layer):
  def __init__(self,
               channels,
               momentum,
               name="point_net_conv"):
    super(PointNetConvLayer, self).__init__(name=name)
    self.channels = channels
    self.momentum = momentum


  def build(self, input_shape):
    """Builds the layer with a specified input_shape."""
    self.conv = tfk.layers.Conv1D(self.channels, 1,input_shape=input_shape)
    self.bn = tf.keras.layers.BatchNormalization(momentum=self.momentum)

  def call(self, inputs, training=None):  # pylint: disable=arguments-differ
    """Executes the convolution.
    Args:
      inputs: a dense tensor of size `[B, N, D]`.
      training: flag to control batch normalization update statistics.
    Returns:
      Tensor with shape `[B, N, C]`.
    """
    return tf.nn.relu(self.bn(self.conv(inputs), training))

class PointNetDenseLayer(tfk.layers.Layer):
  """The fully connected layer used by the classification head in pointnet.
  Note:
    Differently from the standard Keras Conv2 layer, the order of ops is:
      1. fully connected layer
      2. batch normalization layer
      3. ReLU activation unit
  """

  def __init__(self,
               channels,
               momentum,
               name="point_net_dense"):
    super(PointNetDenseLayer, self).__init__(name=name)
    self.momentum = momentum
    self.channels = channels


  def build(self, input_shape):
      self.dense = tfk.layers.Dense(self.channels,input_shape=input_shape)
      self.bn = tfk.layers.BatchNormalization(momentum=self.momentum)

  def call(self, inputs, training=None):  # pylint: disable=arguments-differ
    """Executes the convolution.
    Args:
      inputs: a dense tensor of size `[B, D]`.
      training: flag to control batch normalization update statistics.
    Returns:
      Tensor with shape `[B, C]`.
    """
    return tf.nn.relu(self.bn(self.dense(inputs), training))

class OrthogonalRegularizer(tfk.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

class TransformerNetLayer(tfk.layers.Layer):
  def __init__(self,
               dims,
               l2reg,
               momentum,
               name="transformer_net"):
    super(TransformerNetLayer, self).__init__(name=name)
    self.dims = dims
    self.otg_reg = OrthogonalRegularizer(dims,l2reg)
    self.bias = tfk.initializers.Constant(np.eye(dims).flatten())

    self.conv1 = PointNetConvLayer(16, momentum)
    self.conv2 = PointNetConvLayer(32, momentum)
    self.conv3 = PointNetConvLayer(256, momentum)
    self.dense1 = PointNetDenseLayer(128, momentum)
    self.dense2 = PointNetDenseLayer(64, momentum)
    self.pooling = tfk.layers.GlobalMaxPool1D()
    self.dense3 = tfk.layers.Dense(self.dims * self.dims,
                                  kernel_initializer="zeros",
                                  bias_initializer=self.bias,
                                  activity_regularizer=self.otg_reg)
    self.reshape = tfk.layers.Reshape((self.dims, self.dims))
    self.dot = tfk.layers.Dot((2,1))

  def call(self, inputs, training=None):
    x = self.conv1(inputs, training=training)
    x = self.conv2(x, training=training)
    x = self.conv3(x, training=training)
    x = self.pooling(x)
    x = self.dense1(x, training)
    x = self.dense2(x, training)
    x = self.dense3(x)
    feat_T = self.reshape(x)
    return self.dot([inputs,feat_T])

class PointNet(tfk.Model):
    def __init__(self,
                 class_num=5,
                 name="pointnet_model",
                 momentum=0.5,
                 dropout_rate=0.3,
                 is_vanilla=True,
                 **kwargs):
        super(PointNet, self).__init__(name=name,**kwargs)
        self.is_vanilla = is_vanilla
        self.acc_metric = tfk.metrics.CategoricalAccuracy(name='acc')
        self.conv1 = PointNetConvLayer(16, momentum)
        self.conv2 = PointNetConvLayer(16, momentum)
        self.conv3 = PointNetConvLayer(16, momentum)
        self.conv4 = PointNetConvLayer(32, momentum)
        self.conv5 = PointNetConvLayer(256, momentum)
        self.dense1 = PointNetDenseLayer(128, momentum)
        self.dense2 = PointNetDenseLayer(64, momentum)
        self.dropout = tfk.layers.Dropout(dropout_rate)
        self.dense3 = tfk.layers.Dense(class_num, activation="linear")
        self.pooling = tfk.layers.GlobalMaxPool1D()
        if not is_vanilla:
            self.tnet1 = TransformerNetLayer(dims=3, l2reg=1e-4,momentum=momentum)
            self.tnet2 = TransformerNetLayer(dims=16, l2reg=1e-4, momentum=momentum)

    def call(self,inputs,training=True):

        x = inputs
        if not self.is_vanilla:
            x = self.tnet1(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        if not self.is_vanilla:
            x = self.tnet2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.pooling(x)
        x = self.dense1(x, training)
        x = self.dropout(x, training)
        x = self.dense2(x, training)
        x = self.dropout(x, training)
        logit = self.dense3(x)
        return logit

    @tf.function
    def train_step(self, data):

        inputs, one_hot_label = data
        with tf.GradientTape() as tape:
            logit = self(inputs,training=True)
            loss = self.compiled_loss(one_hot_label, logit, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.acc_metric.update_state(one_hot_label, logit)
        self.compiled_metrics.update_state(one_hot_label, logit)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        inputs, one_hot_label = data
        logit = self(inputs,training=False)
        loss = self.compiled_loss(one_hot_label, logit, regularization_losses=self.losses)
        self.acc_metric.update_state(one_hot_label, logit)
        self.compiled_metrics.update_state(one_hot_label, logit)
        return {m.name: m.result() for m in self.metrics}