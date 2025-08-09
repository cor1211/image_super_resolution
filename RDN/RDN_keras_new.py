from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Concatenate, Lambda, UpSampling2D
from tensorflow.keras.models import Model
import tensorflow as tf

class RDN(Model):
   def __init__(self, arch_params, c_dim=3, kernel_size=3, upscaling='shuffle'):
      super(RDN, self).__init__()
      self.C = arch_params['C']
      self.D = arch_params['D']
      self.G = arch_params['G']
      self.G0 = arch_params['G0']
      self.scale = arch_params['x']
      self.c_dim = c_dim
      self.kernel_size = kernel_size
      self.upscaling = upscaling
      
      # Xây dựng model giống hệt kiến trúc gốc
      LR_input = Input(shape=(None, None, 3), name='LR')
      
      # F_-1
      F_m1 = Conv2D(
         self.G0,
         kernel_size=self.kernel_size,
         padding='same',
         name='F_m1'
      )(LR_input)
      
      # F_0
      F_0 = Conv2D(
         self.G0,
         kernel_size=self.kernel_size,
         padding='same',
         name='F_0'
      )(F_m1)
      
      # RDBs
      rdb_concat = []
      rdb_in = F_0
      for d in range(1, self.D + 1):
         x = rdb_in
         for c in range(1, self.C + 1):
               F_dc = Conv2D(
                  self.G,
                  kernel_size=self.kernel_size,
                  padding='same',
                  name=f'F_{d}_{c}'
               )(x)
               F_dc = Activation('relu', name=f'F_{d}_{c}_Relu')(F_dc)
               x = Concatenate(axis=3, name=f'RDB_Concat_{d}_{c}')([x, F_dc])
         
         x = Conv2D(
               self.G0, 
               kernel_size=1, 
               name=f'LFF_{d}'
         )(x)
         rdb_in = Add(name=f'LRL_{d}')([x, rdb_in])
         rdb_concat.append(rdb_in)
      
      FD = Concatenate(axis=3, name='LRLs_Concat')(rdb_concat)
      
      # Global Feature Fusion
      GFF1 = Conv2D(
         self.G0,
         kernel_size=1,
         padding='same',
         name='GFF_1'
      )(FD)
      
      GFF2 = Conv2D(
         self.G0,
         kernel_size=self.kernel_size,
         padding='same',
         name='GFF_2'
      )(GFF1)
      
      # Global Residual Learning
      FDF = Add(name='FDF')([GFF2, F_m1])
      
      # Upscaling
      if self.upscaling == 'shuffle':
         FU = self._pixel_shuffle(FDF)
      else:
         FU = self._upsampling_block(FDF)
      
      # Final output
      SR = Conv2D(
         self.c_dim,
         kernel_size=self.kernel_size,
         padding='same',
         name='SR'
      )(FU)
      
      self.model = Model(inputs=LR_input, outputs=SR)
   
   def _pixel_shuffle(self, input_layer):
      x = Conv2D(
         self.c_dim * self.scale ** 2,
         kernel_size=3,
         padding='same',
         name='UPN3'
      )(input_layer)
      return Lambda(
         lambda x: tf.nn.depth_to_space(x, block_size=self.scale, data_format='NHWC'),
         name='PixelShuffle'
      )(x)
   
   def _upsampling_block(self, input_layer):
      x = Conv2D(
         self.c_dim * self.scale ** 2,
         kernel_size=3,
         padding='same',
         name='UPN3'
      )(input_layer)
      return UpSampling2D(size=self.scale, name='UPsample')(x)
   
   def call(self, inputs):
      return self.model(inputs)