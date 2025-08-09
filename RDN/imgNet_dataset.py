from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import PIL


class ImageNetDataset(Dataset):
   def __init__(self, root: str, train:bool = True, transform = None):
      self.transform = transform
      self.images_path = []
      self.labels_path = []
      for folder_name in os.listdir(root):
         folder_path = os.path.join(root, folder_name)
         if len(os.listdir(folder_path)) % 2 == 0: # Filter outliers folders
            for iter, image_name in enumerate(os.listdir(folder_path)):
               if train: # Get 0 - > -100
                  if iter >= 100: # Train dataset
                     if 'resized' not in image_name: # Label
                        self.labels_path.append(os.path.join(folder_path, image_name))
                        image_name = image_name.split('.')[0] + '_resized.jpg'
                        self.images_path.append(os.path.join(folder_path, image_name))
               else: # Test
                  if iter < 100:
                     if 'resized' not in image_name: # Label
                        self.labels_path.append(os.path.join(folder_path, image_name))
                        image_name = image_name.split('.')[0] + '_resized.jpg'
                        self.images_path.append(os.path.join(folder_path, image_name))  
   
   def __len__(self):
      return len(self.labels_path)       
   
   def __getitem__(self, index):
      image_path, label_path = self.images_path[index], self.labels_path[index]
      img, label = Image.open(image_path).convert("RGB"), Image.open(label_path).convert("RGB")
      if self.transform:
         img = self.transform(img)
         label = self.transform(label)
      return img, label   
   
if __name__ == "__main__":
   train_dataset = ImageNetDataset(root='imagenet_dataset', train=True, transform=None)
   print(len(train_dataset))
   test_dataset = ImageNetDataset(root = 'imagenet_dataset', train = False, transform=None)
   print(len(test_dataset))