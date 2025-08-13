import torch
import PIL
from PIL import Image
import numpy
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from imgNet_dataset import ImageNetDataset
from RDN import RDN
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
import argparse # Set value for arguments in command line

if __name__ == "__main__":
   if torch.cuda.is_available():
      device = torch.device('cuda:0') # Cuda:0 chooses the first GPU
      # device1 = torch.device('cpu') # CPU
      print(f"Using GPU: {torch.cuda.get_device_name(0)}")
   else:
      device = torch.device('cpu')
      print("No GPU available, using CPU instead.")
   
   # Argument parser for command line arguments
   parser = argparse.ArgumentParser(description='Train RDN model on ImageNet dataset')
   parser.add_argument('--num_epochs', type = int, default = 200, help = 'Number of epochs for training')
   parser.add_argument('--batch_size', type = int, default = 16, help = 'Batch size for training')
   parser.add_argument('--root', type = str, default = '/kaggle/input/dataset-model/imagenet_dataset', help = 'Root directory of the ImageNet dataset')
   parser.add_argument('--num_workers', type = int, default = 2, help = 'Number of workers for DataLoader')
   parser.add_argument('--root_weight_load', type = str, default = '', help = 'Path to model weight')
   parser.add_argument('--scale_factor', type = int, default = 4, help = 'Scale input to output')
   parser.add_argument('--path_weight_save', type = str, default = '', help = 'Path to save model weight')
   
   # Parameters for training
   args = parser.parse_args()
   num_epochs = args.num_epochs
   batch_size = args.batch_size
   root = args.root
   num_workers = args.num_workers
   root_weight = args.root_weight_load
   scale_factor = args.scale_factor
   path_weight_save = args.path_weight_save
   
   transform = Compose(
      transforms=[
         ToTensor(),
      ]
   )

   # Train dataset
   train_dataset = ImageNetDataset(root=root, train=True, transform=transform)
   train_loader = DataLoader(
      dataset=train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      drop_last=False
   )
   
   # Test dataset
   test_dataset = ImageNetDataset(root = root, train = False, transform=transform)
   test_loader = DataLoader(  
      dataset=test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      drop_last=False
   )
   
   num_iter_train = len(train_loader)
   scale_factor=scale_factor
   num_channels=3
   num_features=64
   growth_rate=64
   num_blocks=16
   num_layers=8
   model = RDN(scale_factor=scale_factor, num_channels=num_channels, num_features=num_features, growth_rate=growth_rate, num_blocks=num_blocks, num_layers=num_layers)
   if root_weight != '':
      model.load_state_dict(torch.load(root_weight)) # Load weights
   model.to(device)
   criterion = torch.nn.MSELoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps = 1e-08, weight_decay=0)
   psnr_best = 0 # Initialize best PSNR to save the best psnr model
   psnr = PSNR(data_range=1.0).to(device)
   
   # Training
   for epoch in range(num_epochs):
      # torch.cuda.empty_cache()
      model.train()
      total_loss = 0.0
      tqdm_train_loader = tqdm(train_loader)
      for iter, (img, label) in enumerate(tqdm_train_loader):
         if torch.cuda.is_available():
            img, label = img.to(device), label.to(device)
         outputs = model(img) # Predicted images
         loss_value = criterion(outputs, label) # calculate loss
         total_loss += loss_value.item()
         optimizer.zero_grad() # Zero the gradients
         lv = loss_value.item()
         loss_value.backward() # Backpropagation, compute gradients
         optimizer.step() # Update weights         
         tqdm_train_loader.set_description(f'Epoch [{epoch+1}/{num_epochs}], Iter [{iter+1}/{num_iter_train}], Loss: {lv:.5f}')
         # torch.cuda.empty_cache()
      print(f"Epoch [{epoch+1}/{num_epochs}], Loss_avg: {(total_loss/num_iter_train):.5f}") # Print average loss of epoch
      # Save weights after each epoch
      torch.save(model.state_dict(), f'/{path_weight_save}/rdn_x{scale_factor}-C{num_layers}-D{num_blocks}-G0{num_features}-G{growth_rate}-epoch{epoch+1}.pth') # Save the model weights
      
      # End once epoch -> evaluation
      model.eval()
      psnr_metric = 0.0
      num_iter_test = len(test_loader)
      tqdm_test_loader = tqdm(test_loader)
      with torch.no_grad():
         for iter, (img, label) in enumerate(tqdm_test_loader):
            if torch.cuda.is_available():
               img, label = img.to(device), label.to(device) 
            outputs = model(img)
            psnr_metric += psnr(outputs, label).item() # Calculate PSNR for each batch
            
      # End of test dataset
      psnr_value = psnr_metric/ num_iter_test # Average PSNR over all batches
      print(f'Epoch [{epoch+1}/{num_epochs}], PSNR: {psnr_value:.3f} dB')
      
      # Save the best model based on PSNR
      if psnr_value > psnr_best:
         psnr_best = psnr_value
         psnr_note = str(round(psnr_best, 4)).replace(".", "_")
         torch.save(model.state_dict(), f'/{path_weight_save}/rdn_x{scale_factor}-C{num_layers}-D{num_blocks}-G0{num_features}-G{growth_rate}-best_psnr{psnr_note}.pth') # Save the best model weights
      