from PIL import Image
import os

def crop_image(image: Image.Image, new_size: tuple)->Image.Image:
   croped_img = image.crop(box=new_size)
   return croped_img

if __name__ == '__main__':
   root = r"D:\học\Tài Liệu CNTT\image_super_resolution\dataset\imagenet_64_128_250f"
   new_size = (0,0,128,128)
   
   for folder_name in os.listdir(root): # Iter each folder
      folder_path = os.path.join(root, folder_name)
      for image_name in os.listdir(folder_path): # Iter each image in once folder
         image_path = os.path.join(folder_path, image_name)
         image = Image.open(image_path)
         croped_img = crop_image(image, new_size=new_size)
         croped_img.save(image_path) 
         print(f"Croped {image_path}->{new_size[2]}x{new_size[3]}")        