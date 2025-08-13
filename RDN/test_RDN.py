from RDN import RDN
import torch
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--image_path", type=str, default="", help="Path to the input image")
parser.add_argument("--scale_factor", type=int, default = 2, help="Scale factor for super-resolution")

if __name__ == "__main__":
   args = parser.parse_args()
   path = args.image_path # Path to the input image
   scale_factor = args.scale_factor # Scale-factor
   if not path:
      print("No image path provided.")
      exit(0)
   if torch.cuda.is_available():
      device = torch.device('cuda:0')
      print(f"Using GPU: {torch.cuda.get_device_name(0)}")
   else:
      device = torch.device('cpu')
      print("No GPU available, using CPU instead.")
      
   # Initialize model
   model = RDN(scale_factor=scale_factor, num_channels=3, num_features=64, growth_rate=64, num_blocks=16,num_layers=8)
   model.to(device) # Put model to GPU
   model.load_state_dict(torch.load(r"D:\học\Tài Liệu CNTT\image_super_resolution\weights\image_64_128_250f\rdn_x2-C8-D16-G064-G64-best_psnr26_9196_13.pth"))
   model.eval()  # Set the model to evaluation mode

   image = Image.open(path).convert("RGB") # Open Image from path
   new_size = (image.height* scale_factor, image.width * scale_factor)
   image1 = Resize(new_size, interpolation = PIL.Image.BICUBIC)(image)
   image1.show()
   
   # To tensor from image
   image_tensor = ToTensor()(image).unsqueeze(0)  # Add batch dimension
   if torch.cuda.is_available():
      image_tensor = image_tensor.to(device)
   with torch.no_grad():
      output = model(image_tensor)
   output_image = output.squeeze(0)  # Remove batch dimension
   output_image = output_image.permute(1, 2, 0)  # Change to HWC format
   output_image = (output_image.cpu() * 255).clamp(0, 255).byte().numpy()  # Convert to uint8
   # print(output_image)
   output_image = Image.fromarray(output_image)
   output_image.show()