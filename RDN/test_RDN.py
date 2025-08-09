from RDN import RDN
import torch
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import matplotlib.pyplot as plt

if __name__ == "__main__":
   model = RDN(scale_factor=2, num_channels=3, num_features=64, growth_rate=64, num_blocks=16,num_layers=8)
   model.load_state_dict(torch.load(r'D:\học\Tài Liệu CNTT\image_super_resolution\weights\image_64_128_250f\rdn_x2-C8-D16-G064-G64-best_psnr26_7808.pth'))
   model.eval()  # Set the model to evaluation mode

   path = r"D:\học\Tài Liệu CNTT\image_super_resolution\dataset\imagnet_256_full_origin\zucchini\506.jpg"
   image = Image.open(path).convert("RGB")
   # print(f"{np.array(image)}\n--------------------")
   image1 = Resize((512, 512), interpolation=PIL.Image.LANCZOS)(image)
   
   image1.show()
   
   image_tensor = ToTensor()(image).unsqueeze(0)  # Add batch dimension
   # image_tensor = pil_to_tensor_255(image).unsqueeze(0) # Add batch dimension

   with torch.no_grad():
      output = model(image_tensor)
   output_image = output.squeeze(0)  # Remove batch dimension
   output_image = output_image.permute(1, 2, 0)  # Change to HWC format
   output_image = (output_image * 255).clamp(0, 255).byte().numpy()  # Convert to uint8
   # print(output_image)
   output_image = Image.fromarray(output_image)
   output_image.show()