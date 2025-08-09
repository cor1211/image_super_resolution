import os
import PIL
from PIL import Image

def save_image_1(root, image:Image.Image, img_name:str)->None:
   new_img_name = img_name.split('.')[0] + '_resized.bmp'
   new_path = os.path.join(root, new_img_name)
   image.save(new_path)
   print(f"Image saved at {new_path}")

def save_image_2(folder_path,folder_name, img_name, image:Image.Image) -> None:
   new_img_name = folder_name + img_name.split('.')[0] + '_resized.jpg'
   new_path = os.path.join(folder_path, new_img_name)
   image.save(new_path)
   print(f"Image saved at {new_path}")

def save_image_3(folder_path, img_name, image:Image.Image) -> None:
   new_img_name = img_name.split('.')[0] + '_resized.jpg'
   new_path = os.path.join(folder_path, new_img_name)
   image.save(new_path)
   print(f"Image saved at {new_path}")
   
   
   
def img_resize1(root: str, new_size: tuple) -> None:
   for image_name in os.listdir(root):
      image_path = os.path.join(root, image_name)
      with Image.open(image_path) as img:
         resized_img = img.resize(new_size)
         save_image_1(root, resized_img, image_name)

def img_resized2(root:str, new_size: tuple) -> None:
   for folder_name in os.listdir(root):
      folder_path = os.path.join(root, folder_name)
      for image_name in os.listdir(folder_path):
         image_path = os.path.join(folder_path, image_name)
         with Image.open(image_path) as img:
            resized_img = img.resize(size=new_size)
            save_image_3(folder_path, image_name, resized_img)
            
if __name__ == "__main__":
   # img_resize1(root='dataset/test', new_size=(128, 128))
   
   img_resized2(root=r'D:\học\Tài Liệu CNTT\image_super_resolution\dataset\imagenet_small_64_256_2f_testcode', new_size=(64,64))