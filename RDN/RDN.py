import torch
from torch import nn

class RDB(nn.Module):
   def __init__(self, in_channels, growth_rate, num_layers):
      super().__init__()
      self.num_layers = num_layers
      self.conv_layers = nn.ModuleList()
      for i in range(self.num_layers):
         in_ = in_channels + i * growth_rate
         conv = nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=growth_rate, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
         )
         self.conv_layers.append(conv)
      
      self.lff = nn.Conv2d(in_channels=in_channels + self.num_layers * growth_rate, out_channels= in_channels, kernel_size=1)
   def forward(self, x):
      identity = x
      features = [identity]
      for i in range(self.num_layers):
         # Concat the input of block with output of previous layers
         concatnated_features = torch.cat(features, dim=1)
         out = self.conv_layers[i](concatnated_features)
         features.append(out)
      all_features = torch.cat(features, dim = 1)
      fused_features = self.lff(all_features)
      return identity + fused_features
   

class RDN(nn.Module):
   def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
      """
        Khởi tạo mô hình RDN hoàn chỉnh.
        Args:
            scale_factor (int): Hệ số phóng đại (2, 3, hoặc 4).
            num_channels (int): Số kênh của ảnh đầu vào (ví dụ: 3 cho RGB).
            num_features (int): Số lượng kênh đặc trưng cơ bản (G0).
            growth_rate (int): Tốc độ tăng trưởng trong mỗi RDB (G).
            num_blocks (int): Số lượng khối RDB (D).
            num_layers (int): Số lượng lớp conv trong mỗi RDB (C).
      """
      super().__init__()
      # Part 1: SFENet
      self.sfe1 = nn.Conv2d(in_channels=num_channels, out_channels=num_features, kernel_size=3, padding=1)
      self.sfe2 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)
      
      # Part 2: RDBs
      self.rdbs = nn.ModuleList()
      for _ in range(num_blocks):
         self.rdbs.append(RDB(in_channels=num_features, growth_rate=growth_rate, num_layers=num_layers)) 
      
      # Part 3: DFF
      self.gff = nn.Sequential(
         nn.Conv2d(in_channels=num_blocks*num_features, out_channels=num_features, kernel_size=1, padding=0),
         nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)
      )
      
      # Part 4: UPNet
      self.upscale = nn.Sequential(
         nn.Conv2d(in_channels=num_features, out_channels=num_features * (scale_factor ** 2), kernel_size=3, padding=1),
         nn.PixelShuffle(upscale_factor=scale_factor),
         nn.Conv2d(in_channels=num_features, out_channels=num_channels, kernel_size=3, padding=1)
      )
   
   def forward(self, x) -> torch.Tensor:
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
      gff_in = torch.cat(rdbs_out, dim=1)
      gff_out = self.gff(gff_in)
      
      # GRL (Global Residual Learning): Cộng đặc trưng toàn cục với đặc trưng nông ban đầu
      grl_out = gff_out + sfe1_out
      
      # UPNet
      output = self.upscale(grl_out)
      return output

if __name__ == '__main__':
   scale = 3
   model = RDN(scale_factor=scale, num_channels=3, num_features=64, growth_rate=64,num_blocks=16, num_layers=8 )
   input_tensor = torch.rand(1, 3, 32, 32)
   output_tensor = model(input_tensor)
   # In ra kích thước đầu ra
   print(f"Kích thước đầu vào: {input_tensor.shape}")
   print(f"Kích thước đầu ra: {output_tensor.shape}")
   print(f"Kích thước đầu ra mong đợi: (1, 3, {32*scale}, {32*scale})")