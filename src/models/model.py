import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
  def __init__(self, filters):
    super(Encoder, self).__init__()
    
    self.sep1 = tf.keras.layers.SeparableConv2D(int(filters / 4), [5,5], strides=[1,1], padding='same', activation='relu')
    self.sep2 = tf.keras.layers.SeparableConv2D(filters, [5,5], strides=[1,1], padding='same')

  def call(self, input):
    print('input for encoder:', input.shape)
    output = self.sep1(input)
    output = self.sep2(output)
    print('output for encoder:', output.shape + input.shape)

    return output + input

class Downsample(tf.keras.layers.Layer):
  def __init__(self, filters):
    super(Downsample, self).__init__()

    self.sep1 = tf.keras.layers.SeparableConv2D(int(filters / 4), [5,5], strides=[2,2], padding='same', activation='relu')
    self.sep2 = tf.keras.layers.SeparableConv2D(filters, [5,5], strides=[1,1], padding='same')
    self.sep3 = tf.keras.layers.SeparableConv2D(filters, [3,3], strides=[2,2], padding='same')
  
  def call(self, input):
    print('input for downsample:', input.shape)
    out1 = self.sep1(input)
    out1 = self.sep2(out1)

    out2 = self.sep3(input)
    print('output for downsample:', out1.shape + out2.shape)    
    return out1 + out2

class Decoder(tf.keras.layers.Layer):
  def __init__(self, filters):
    super(Decoder, self).__init__()

    self.sep1 = tf.keras.layers.SeparableConv2D(filters, [3,3], strides=[1,1], padding='same', activation='relu')
    self.sep2 = tf.keras.layers.SeparableConv2D(filters, [3,3], strides=[1,1], padding='same')

  def call (self, input):
    print('input for decoder:', input.shape)
    out1 = self.sep1(input)
    out1 = self.sep2(out1)
    print('output for decoder:', out1.shape + input.shape)
    return out1 + input

class Upsample(tf.keras.layers.Layer):
  def __init__(self, filters):
    super(Upsample, self).__init__()
    self.deconv = tf.keras.layers.Conv2DTranspose(filters, (3,3), strides=2, padding='same')

  def call (self, input):
    print('input for upsample:', input.shape)
    output = self.deconv(input)
    print('output for upsample:', output.shape)
    return output  

class DenoiseNetwork(tf.keras.Model):
  def __init__(self):
    super(DenoiseNetwork, self).__init__()

    # Input stage
    self.input_stage = tf.keras.layers.Conv2D(16, (3,3), padding="same", input_shape=(160,120,1))

    # Encoder Stage 1
    self.encoder_stage1 = tf.keras.Sequential([
        Downsample(64),
        Encoder(64)                       
    ])

    # Encoder Stage 2
    self.encoder_stage2 = tf.keras.Sequential([
        Downsample(128),
        Encoder(128)                       
    ])

    # Encoder Stage 3
    self.encoder_stage3 = tf.keras.Sequential([
        Downsample(256),
        Encoder(256),
        Encoder(256),
        Encoder(256)                      
    ])

    # Encoder Stage 4
    self.encoder_stage4 = tf.keras.Sequential([
        Downsample(512),
        Encoder(512),
        Encoder(512),
        Encoder(512)                      
    ])

    # Decoder Stage 1
    self.decoder_stage1 = tf.keras.Sequential([
        Decoder(512),
        Upsample(64)
    ])

    # Decoder Stage 2
    self.decoder_stage2 = tf.keras.Sequential([
        Decoder(64),
        Upsample(32)
    ])
    # Decoder Stage 3
    self.decoder_stage3 = tf.keras.Sequential([
        Decoder(32),
        Upsample(32)
    ])
    # Decoder Stage 4
    self.decoder_stage4 = tf.keras.Sequential([
        Decoder(32),
        Upsample(16)
    ])

    # Separatable Convolution
    self.sep1 = tf.keras.layers.SeparableConv2D(16, (3,3), padding='same')
    self.sep2 = tf.keras.layers.SeparableConv2D(32, (3,3), padding='same')
    self.sep3 = tf.keras.layers.SeparableConv2D(32, (3,3), padding='same')
    self.sep4 = tf.keras.layers.SeparableConv2D(64, (3,3), padding='same')

    # Output Stage
    self.output_stage = tf.keras.Sequential([
        Decoder(16),
        tf.keras.layers.Conv2D(1, kernel_size=(3,3), padding='same')
    ])

  def call(self, input):

    out_i = self.input_stage(input)
    out_e1 = self.encoder_stage1(out_i)
    out_e2 = self.encoder_stage2(out_e1)
    out_e3 = self.encoder_stage3(out_e2)
    out_e4 = self.encoder_stage4(out_e3)

    # out_d1 = self.decoder_stage1(out_e4) + self.sep4(out_e3)
    decoder_out = self.decoder_stage1(out_e4)
    sep_out = self.sep4(out_e3)
    print('decoder_out:', decoder_out.shape)
    print('sep_out:', sep_out.shape)
    out_d1 = decoder_out + sep_out
    out_d2 = self.decoder_stage2(out_d1) + self.sep3(out_e2)
    out_d3 = self.decoder_stage3(out_d2) + self.sep2(out_e1)
    out_d4 = self.decoder_stage4(out_d3) + self.sep1(out_i)
    output = self.output_stage(out_d4) + input

    return output