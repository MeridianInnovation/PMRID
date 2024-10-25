import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
  def __init__(self, filters):
    super(Encoder, self).__init__()
    
    self.sep1 = tf.keras.layers.SeparableConv2D(int(filters / 4), [5,5], strides=[1,1], padding='same', activation='relu')
    self.sep2 = tf.keras.layers.SeparableConv2D(filters, [5,5], strides=[1,1], padding='same')

  def call(self, input):
    output = self.sep1(input)
    output = self.sep2(output)

    return tf.keras.activations.relu(output + input)

class Downsample(tf.keras.layers.Layer):
  def __init__(self, filters):
    super(Downsample, self).__init__()

    self.sep1 = tf.keras.layers.SeparableConv2D(int(filters / 4), [5,5], strides=[2,2], padding='same', activation='relu')
    self.sep2 = tf.keras.layers.SeparableConv2D(filters, [5,5], strides=[1,1], padding='same')
    self.sep3 = tf.keras.layers.SeparableConv2D(filters, [3,3], strides=[2,2], padding='same')
  
  def call(self, input):
    out1 = self.sep1(input)
    out1 = self.sep2(out1)

    out2 = self.sep3(input)
    
    return out1 + out2

class Decoder(tf.keras.layers.Layer):
  def __init__(self, filters):
    super(Decoder, self).__init__()

    self.sep1 = tf.keras.layers.SeparableConv2D(filters, [3,3], strides=[1,1], padding='same', activation='relu')
    self.sep2 = tf.keras.layers.SeparableConv2D(filters, [3,3], strides=[1,1], padding='same')

  def call (self, input):
    out1 = self.sep1(input)
    out1 = self.sep2(out1)

    return out1 + input

class Upsample(tf.keras.layers.Layer):
  def __init__(self, filters):
    super(Upsample, self).__init__()
    self.deconv = tf.keras.layers.Conv2DTranspose(filters, (3,3), strides=2, padding='same')

  def call (self, input):
    output = self.deconv(input)

    return output  

class DenoiseNetwork(tf.keras.Model):
  def __init__(self, **kwargs):
    super(DenoiseNetwork, self).__init__(**kwargs)

    # Input stage
    self.input_stage = tf.keras.layers.Conv2D(16, (3,3), padding="same")

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

    # # Encoder Stage 4
    # self.encoder_stage4 = tf.keras.Sequential([
    #     Downsample(512),
    #     Encoder(512),
    #     Encoder(512),
    #     Encoder(512)                      
    # ])

    # # Decoder Stage 1
    # self.decoder_stage1 = tf.keras.Sequential([
    #     Decoder(512),
    #     Upsample(64)
    # ])

    # Decoder Stage 1
    self.decoder_stage1 = tf.keras.Sequential([
        Decoder(256),
        Upsample(32)
    ])
    # Decoder Stage 3
    self.decoder_stage2 = tf.keras.Sequential([
        Decoder(32),
        Upsample(32)
    ])
    # Decoder Stage 4
    self.decoder_stage3 = tf.keras.Sequential([
        Decoder(32),
        Upsample(16)
    ])

    # Separatable Convolution
    self.sep1 = tf.keras.layers.SeparableConv2D(16, (3,3), padding='same')
    self.sep2 = tf.keras.layers.SeparableConv2D(32, (3,3), padding='same')
    self.sep3 = tf.keras.layers.SeparableConv2D(32, (3,3), padding='same')
    # self.sep4 = tf.keras.layers.SeparableConv2D(64, (3,3), padding='same')

    # Output Stage
    self.output_stage = tf.keras.Sequential([
        Decoder(16),
        tf.keras.layers.Conv2D(1, kernel_size=(3,3), padding='same')
    ])

  def get_config(self):
    return super(DenoiseNetwork, self).get_config()
  
  def call(self, input):

    out_i = self.input_stage(input)
    out_e1 = self.encoder_stage1(out_i)
    out_e2 = self.encoder_stage2(out_e1)
    out_e3 = self.encoder_stage3(out_e2)
    # out_e4 = self.encoder_stage4(out_e3)

    # out_d1 = self.decoder_stage1(out_e4) + self.sep4(out_e3)
    out_d1 = self.decoder_stage1(out_e3) + self.sep3(out_e2)
    out_d2 = self.decoder_stage2(out_d1) + self.sep2(out_e1)
    out_d3 = self.decoder_stage3(out_d2) + self.sep1(out_i)
    output = self.output_stage(out_d3) + input

    return output
  
if __name__ == '__main__':
  net = DenoiseNetwork()
  img = tf.random.normal((1, 160, 120, 1))
  out = net(img)

  print("Input Shape:", img.shape)
  print("Output Shape:", out.shape)
  print("Output Values:", out.numpy())
  print("Output Mean:", tf.reduce_mean(out).numpy())
  print("Output Standard Deviation:", tf.math.reduce_std(out).numpy())
  print("Output Min:", tf.reduce_min(out).numpy())
  print("Output Max:", tf.reduce_max(out).numpy())

