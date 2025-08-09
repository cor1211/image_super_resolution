import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.layers import Conv2D, ReLU, Concatenate, Lambda
from keras.models import Sequential


class RDB(layers.Layer):
   def __init__(self, in_channels, growth_rate, num_layers):
      super(RDB, self).__init__()
      self.num_layers = num_layers
      self.conv_layers = []
      for i in range(self.num_layers):
         conv = Sequential([
               Conv2D(growth_rate, kernel_size=3, padding='same'),
               ReLU()
         ])
         self.conv_layers.append(conv)
      
      self.lff = Conv2D(in_channels, kernel_size=1)
   
   def call(self, x):
      identity = x
      features = [identity]
      for i in range(self.num_layers):
         # Concat the input of block with output of previous layers
         concatnated_features = Concatenate(axis=-1)(features)
         out = self.conv_layers[i](concatnated_features)
         features.append(out)
      all_features = Concatenate(axis=-1)(features)
      fused_features = self.lff(all_features)
      return layers.add([identity, fused_features])

class RDN(Model):
   def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
      """
      Initialize the complete RDN model.
      Args:
         scale_factor (int): Upscaling factor (2, 3, or 4).
         num_channels (int): Number of input image channels (e.g., 3 for RGB).
         num_features (int): Number of base feature channels (G0).
         growth_rate (int): Growth rate in each RDB (G).
         num_blocks (int): Number of RDB blocks (D).
         num_layers (int): Number of conv layers in each RDB (C).
      """
      super(RDN, self).__init__()
      # Part 1: SFENet
      self.sfe1 = Conv2D(num_features, kernel_size=3, padding='same')
      self.sfe2 = Conv2D(num_features, kernel_size=3, padding='same')
      
      # Part 2: RDBs
      self.rdbs = []
      for _ in range(num_blocks):
         self.rdbs.append(RDB(in_channels=num_features, growth_rate=growth_rate, num_layers=num_layers)) 
      
      # Part 3: DFF
      self.gff = Sequential([
         Conv2D(num_blocks*num_features, kernel_size=1, padding='valid'),
         Conv2D(num_features, kernel_size=3, padding='same')
      ])
      
      # Part 4: UPNet
      self.upscale = Sequential([
         Conv2D(num_features * (scale_factor ** 2), kernel_size=3, padding='same'),
         Lambda(lambda x: tf.nn.depth_to_space(x, scale_factor)),
         Conv2D(num_channels, kernel_size=3, padding='same')
      ])
   
   def call(self, x):
      # SFENet
      sfe1_out = self.sfe1(x)
      sfe2_out = self.sfe2(sfe1_out)
      
      # RDBs
      rdbs_out = []
      in_features = sfe2_out
      for rdb in self.rdbs:
         rdb_out = rdb(in_features)
         rdbs_out.append(rdb_out)
         in_features = rdb_out
      
      # DFF
      # GFF
      gff_in = Concatenate(axis=-1)(rdbs_out)
      gff_out = self.gff(gff_in)
      
      # GRL (Global Residual Learning): Add global features with initial shallow features
      grl_out = layers.add([gff_out, sfe1_out])
      
      # UPNet
      output = self.upscale(grl_out)
      return output

if __name__ == '__main__':
   scale = 3
   model = RDN(scale_factor=scale, num_channels=3, num_features=64, 
               growth_rate=64, num_blocks=16, num_layers=8)
   
   # Build model by calling it on a sample input
   input_tensor = tf.random.normal((1, 32, 32, 3))
   output_tensor = model(input_tensor)
   
   # Print model summary
   model.summary()
   
   # Print input/output shapes
   print(f"Input shape: {input_tensor.shape}")
   print(f"Output shape: {output_tensor.shape}")
   print(f"Expected output shape: (1, {32*scale}, {32*scale}, 3)")