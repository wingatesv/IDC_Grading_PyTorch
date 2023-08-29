import torch 
from torch import nn
import torchvision
import zipfile
import pathlib 
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import PIL
import torch.utils.data as data
from sklearn.utils import shuffle
import sklearn.model_selection
from typing import Tuple, Dict, List
from torch.utils.data import Dataset
import staintools

from StainNet.models import StainNet, ResnetGenerator

def get_file_paths(dir_path):
  file_paths = []
  for dir_path, dirnames, filenames in os.walk(dir_path):
    for filename in filenames:
      file_path = os.path.join(dir_path, filename)
      file_paths.append(file_path)
  return file_paths

def find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
  classes = sorted(entry.name for entry in os.scandir(dir) if entry.is_dir())

  if not classes:
    raise FileNotFoundError(f'Couldnt find any classes in {dir}... pleases check file structure')
  
  class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
  return classes, class_to_idx

def norm(image):
    image = np.array(image).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = ((image / 255) - 0.5) / 0.5
    image=image[np.newaxis, ...]
    image=torch.from_numpy(image)
    return image

def un_norm(image):
    image = image.cpu().detach().numpy()[0]
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8).transpose((1,2,0))
    return image

class CSVImageDataset(data.Dataset):
    def __init__(self, csv_file, transform=None, sn=None, temp_dir = None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sn = sn

        if self.sn == 'reinhard':
          self.normalizer  = staintools.ReinhardColorNormalizer()
          self.normalizer.fit(staintools.read_image(temp_dir))
        elif self.sn in ['macenko','vahadane']:
          self.normalizer = staintools.StainNormalizer(method=self.sn)
          self.normalizer.fit(staintools.read_image(temp_dir))
        elif self.sn == 'stainnet':
          # load  pretrained StainNet
          self.net = StainNet().to(self.device)
          self.net.load_state_dict(torch.load("/content/drive/Shareddrives/Drive/PhD/IDC_Grading_PyTorch/StainNet/checkpoints/camelyon16_dataset/StainNet-Public-centerUni_layer3_ch32.pth", map_location=torch.device(self.device)))
        elif self.sn == 'staingan':
          #load  pretrained StainGAN
          self.net = ResnetGenerator(3, 3, ngf=64, norm_layer=torch.nn.InstanceNorm2d, n_blocks=9).to(self.device)
          self.net.load_state_dict(torch.load("/content/drive/Shareddrives/Drive/PhD/IDC_Grading_PyTorch/StainNet/checkpoints/camelyon16_dataset/latest_net_G_A.pth", map_location=torch.device(self.device)))
          
        elif self.sn == 'none':
          pass  
        else:
          raise Exception("Please specify the correct stain normalisation method")


    def __len__(self):
        return len(self.labels_frame)

    def load_img(self, index: int) -> PIL.Image.Image:
      # opens an image via PIL and returns it

      img_path = self.labels_frame.iloc[index,1]
  

      if self.sn == 'none':
        img = PIL.Image.open(img_path)
        return img
      elif self.sn in ['stainnet', 'staingan']:
        img = PIL.Image.open(img_path)
        img=self.net(norm(img).to(self.device))
        img=un_norm(img)
        return img
      else:
        img = staintools.read_image(str(img_path))
        return self.normalizer.transform(img)

    def __getitem__(self, idx):
        image = self.load_img(idx)
        # img_path = self.labels_frame.iloc[idx,1]
        # image = PIL.Image.open(img_path)
        label = int(self.labels_frame.iloc[idx, 2])

        if self.transform:
            image = self.transform(image)

        return image, label



class CustomDataset(Dataset):
  def __init__(self, root, transform = None, sn = None, temp_dir = None):
    super().__init__()
    self.root = root
    self.paths = list(pathlib.Path(root).glob('*/*')) #need chg the file type
    self.transform = transform
    self.classes, self.class_to_idx = find_classes(root)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.sn = sn

    if self.sn == 'reinhard':
      self.normalizer  = staintools.ReinhardColorNormalizer()
      self.normalizer.fit(staintools.read_image(temp_dir))
    elif self.sn in ['macenko','vahadane']:
      self.normalizer = staintools.StainNormalizer(method=self.sn)
      self.normalizer.fit(staintools.read_image(temp_dir))
    elif self.sn == 'stainnet':
      # load  pretrained StainNet
      self.net = StainNet().to(self.device)
      self.net.load_state_dict(torch.load("/content/drive/Shareddrives/Drive/PhD/IDC_Grading_PyTorch/StainNet/checkpoints/camelyon16_dataset/StainNet-Public-centerUni_layer3_ch32.pth", map_location=torch.device(self.device)))
    elif self.sn == 'staingan':
      #load  pretrained StainGAN
      self.net = ResnetGenerator(3, 3, ngf=64, norm_layer=torch.nn.InstanceNorm2d, n_blocks=9).to(self.device)
      self.net.load_state_dict(torch.load("/content/drive/Shareddrives/Drive/PhD/IDC_Grading_PyTorch/StainNet/checkpoints/camelyon16_dataset/latest_net_G_A.pth", map_location=torch.device(self.device)))
      
    elif self.sn == 'none':
      pass  
    else:
      raise Exception("Please specify the correct stain normalisation method")

  def load_img(self, index: int) -> PIL.Image.Image:
    # opens an image via PIL and returns it
    img_path = self.paths[index]
 

    if self.sn == 'none':
      img = PIL.Image.open(img_path)
      return img
    elif self.sn in ['stainnet', 'staingan']:
      img = PIL.Image.open(img_path)
      img=self.net(norm(img).to(self.device))
      img=un_norm(img)
      return img
    else:
      img = staintools.read_image(str(img_path))
      return self.normalizer.transform(img)

  def __len__(self) -> int:
    # returns the total number of images
    return len(self.paths)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
    # returns one sample of data and label in tuple
    img = self.load_img(index)
    class_name = self.paths[index].parent.name #expects path in format: data_folder/class_name/imge.jpg
    class_idx = self.class_to_idx[class_name]

    # transform if necessary
    if self.transform:
        return self.transform(img), class_idx
    else:
      return img, class_idx

def create_df(file_paths, return_label = True):
  df= pd.DataFrame(index=np.arange(0, len(file_paths)), columns=["path", "grade"])

  for idx, image_path in enumerate(file_paths):
    path_name = str(image_path)
    grade = str(image_path.split("/")[-2]).rstrip()
    # only take the grade number
    grade = grade[-1]
    df.iloc[idx]["path"] = path_name
    df.iloc[idx]["grade"] = grade
  
  df = shuffle(df)

  if return_label:
    return df, df[['grade']] 
  else:
    return df

def generate_class_weights(data):
  labels = [label for _, label in data]
  # get class weights
  class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(labels), y = labels)
  class_weights = torch.tensor(class_weights, dtype=torch.float)
  return class_weights

def unzip_file(data_path, zip_path, unzip):
  if unzip :
    # create directory if not exist
    if not os.path.isdir(data_path):
      os.makedirs(data_path)

    # check dir exist and empty
    if os.path.isdir(data_path) and len(os.listdir(data_path)) == 0:
      with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print('unzipping data...')
        zip_ref.extractall(data_path)
    else:
      print(f'skipping unzipping {zip_path}.....directory exists')



# only generate train set and test set
def FBCG(data_path, zip_path, batch_size, augmentation = True, unzip = True, test_mode = False, sn = None, temp_dir = None):
  # unzip file if required
  unzip_file(data_path, zip_path, unzip)

  # setup training and test paths
  train_dir = data_path + '/FBCG Dataset/Training Set' 
  test_dir = data_path + '/FBCG Dataset/Test Set'

  # data augmentation if set true 
  if augmentation:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize(size=(224,224)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=72), ])

  else:
    train_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      transforms.Resize(size=(224,224)),])


  test_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      transforms.Resize(size=(224,224)) ])



  # create dataloader from train_dir and test_dir
  train_data= CustomDataset(root=train_dir, transform=train_transform, sn = sn, temp_dir = temp_dir)
  # get class_weights
  class_weights = generate_class_weights(train_data)

  # create dataloader from test dir
  test_data =CustomDataset(root=test_dir, transform=test_transform, sn = sn, temp_dir = temp_dir)
  train_dataloader = DataLoader(dataset = train_data, batch_size = batch_size, num_workers=os.cpu_count(), shuffle = True)
  test_dataloader = DataLoader(dataset = test_data, batch_size = batch_size, num_workers=os.cpu_count(), shuffle = False)

  if not test_mode:
    return train_dataloader, test_dataloader, class_weights
  else:
    return test_dataloader, torch.Tensor(test_data.targets)


#  generate k-fold split 
def FBCG_cv(data_path, zip_path, batch_size, augmentation = True, unzip = True, split = 5, fold = 1, sn=None, temp_dir=None):
  # unzip file if required
  if fold == 1:
    unzip_file(data_path, zip_path, unzip)

  # setup training and test paths
  train_dir = data_path + '/FBCG Dataset/Training Set' 

  # create directory to store csv file
  result_dir = os.path.join(data_path, 'cv_fold')
  if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

  # if csv files are not found
  if os.path.isdir(result_dir) and len(os.listdir(result_dir)) == 0:
    print('No CSV files found...making SFFCV files')
    # get all file paths
    file_paths = get_file_paths(train_dir)
    # generate df and labels form file_paths
    df, label = create_df(file_paths, return_label=True)

    # initialise stratifield k-fold
    skf = sklearn.model_selection.StratifiedKFold (n_splits = split, random_state = 123, shuffle = True)

    # k split and generate csv files
    idx = 1
    for train_index, val_index in skf.split(np.zeros(len(df)), label):
      training_data = df.iloc[train_index]
      validation_data = df.iloc[val_index]

      training_data.to_csv(result_dir + f'/training_data_{idx}.csv')
      validation_data.to_csv(result_dir + f'/validation_data_{idx}.csv')
      idx +=1
  else:
    print('CSV files found....skipping')
  
  #  initialise k fold
  cur_csv_train_dir = result_dir + f'/training_data_{fold}.csv'
  cur_csv_val_dir = result_dir + f'/validation_data_{fold}.csv'

  # data augmentation if set true 
  if augmentation:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize(size=(224,224)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=72)
        ])
  else:

    train_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      transforms.Resize(size=(224,224))
      ])

  val_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      transforms.Resize(size=(224,224)),
      

  ])

  #  create pytorch data
  train_data = CSVImageDataset(cur_csv_train_dir, transform = train_transform, sn = sn, temp_dir = temp_dir)
  val_data = CSVImageDataset(cur_csv_train_dir, transform = val_transform, sn = sn, temp_dir = temp_dir)

  #  generate class_weights
  class_weights = generate_class_weights(train_data)
  #  create dataloader
  train_dataloader = DataLoader(dataset = train_data, batch_size = batch_size, num_workers=os.cpu_count(), shuffle = True)
  val_dataloader = DataLoader(dataset = val_data, batch_size = batch_size, num_workers=os.cpu_count(), shuffle = False)

  return train_dataloader, val_dataloader, class_weights




