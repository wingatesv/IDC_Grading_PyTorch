import torch 
import os

# import from other python files
import config
from io_utils import parse_args, get_resume_file
from model import IDC_Grading_Model, train_step, val_step
from dataset import FBCG, FBCG_cv
from get_temp import get_temp

def model_train(start_epoch, stop_epoch, train_dataloader, val_dataloader, model, loss_fn, optimizer, device):
   # initialise max accuracy to check for best performing model weights
  max_acc = 0   
 
  # Training loop
  for epoch in range(start_epoch, stop_epoch):
      print(f"Epoch {epoch+1}\n-------------------------------")

      # perform train step (forward batches, compute loss, backpropagation, gradient steop)
      train_step(train_dataloader, model, loss_fn, optimizer, device)
      # check model performance 
      acc = val_step(val_dataloader, model, loss_fn, device)

      # if acc higher than max acc
      if acc > max_acc : 
          print("best model! save...")
          max_acc = acc
          outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
          # save best model weights
          torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

      # save model weight every save_freq or last epoch
      if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
          outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
          torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)



if __name__=='__main__':
  
  # setup device agnostic code
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # get overall and training parsing arguments
  params = parse_args('train')

  # initialise training parsing arguments
  feature_extractor = params.feature_extractor
  unzip = params.unzip
  train_aug = params.train_aug
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  learning_rate = params.lr
  batch_size = params.batch_size

  # setup templates image path
  if params.sn in ['reinhard', 'macenko', 'vahadane']:
    # get temp images if required
    get_temp(config.DATA_PATH)
    if params.temp in ['Temp1', 'Temp2', 'Temp3', 'Temp4', 'Temp5']:
      temp_dir = config.TEMP_DIR + f'/{params.temp}.png'
    else:
      raise Exception('wrong template image')
  else:
    temp_dir = None
    params.temp = 'None'



  # perform normal model training
  if not params.cross_val:
    print(f'Training {feature_extractor} classifier with lr = {learning_rate} for {stop_epoch} epochs in {device} with {params.sn}/{params.temp}')

    # create checkpoint directory to save best model
    params.checkpoint_dir = '%s/checkpoints/%s' %(config.SAVE_DIR, params.feature_extractor)
    # add special notation for training augmentation
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    # if stain normalisation is applied
    if params.sn != 'none':
       params.checkpoint_dir += f'_{params.sn}'
       params.checkpoint_dir += f'_{params.temp}'

    # create directory if not exist
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # get FBCG training and val dataloaders (val dataloader is from test set)
    train_dataloader, val_dataloader, class_weights = FBCG(data_path = config.DATA_PATH, zip_path = config.ZIP_PATH, augmentation = train_aug, unzip = unzip, batch_size=batch_size, sn = params.sn, temp_dir = temp_dir)

    # initialise CNN model pass to device
    model = IDC_Grading_Model(feature_extractor, class_weights).to(device)

    # initialise cost function
    loss_fn = model.loss_fn

    # get model parameters based on diferent feature extractor
    if feature_extractor == 'resnet50':
      model_parameters = model.feature_extractor.fc.parameters()
    else:
      model_parameters = model.feature_extractor.classifier.parameters()

    # initialise training optimizer with specified learning rate
    optimizer = torch.optim.Adam(model_parameters, lr=learning_rate)

    # resume model training
    if params.resume:
      resume_file = get_resume_file(params.checkpoint_dir)
      if resume_file is not None:
          tmp = torch.load(resume_file)
          start_epoch = tmp['epoch']+1
          model.load_state_dict(tmp['state'])
          print(f'{feature_extractor} trained weights loaded')
    
    model_train(start_epoch, stop_epoch, train_dataloader, val_dataloader, model, loss_fn, optimizer, device)

  # perform SFFCV
  else:
    
    split = params.ksplit
    for fold in range(1,split+1):
      print(f'Training {feature_extractor} classifier with lr = {learning_rate} for {stop_epoch} epochs in {device} at {fold}/{split} fold with {params.sn}/{params.temp}')

      # create checkpoint directory to save best model
      params.checkpoint_dir = '%s/checkpoints/%s_cv_%s' %(config.SAVE_DIR, params.feature_extractor, fold)
      # add special notation for training augmentation
      if params.train_aug:
          params.checkpoint_dir += '_aug'
      
      # if stain normalisation is applied
      if params.sn != 'none':
        params.checkpoint_dir += f'_{params.sn}'
        params.checkpoint_dir += f'_{params.temp}'

      # create directory if not exist
      if not os.path.isdir(params.checkpoint_dir):
          os.makedirs(params.checkpoint_dir)

      # get train and val set from each fold
      train_dataloader, val_dataloader, class_weights = FBCG_cv(data_path = config.DATA_PATH, zip_path = config.ZIP_PATH, augmentation = train_aug, unzip = unzip, batch_size=batch_size, split = split, fold = fold, sn = params.sn, temp_dir = temp_dir)

      # initialise CNN model pass to device
      model = IDC_Grading_Model(feature_extractor, class_weights).to(device)

      # initialise cost function
      loss_fn = model.loss_fn

      # get model parameters based on diferent feature extractor
      if feature_extractor == 'resnet50':
        model_parameters = model.feature_extractor.fc.parameters()
      else:
        model_parameters = model.feature_extractor.classifier.parameters()

      # initialise training optimizer with specified learning rate
      optimizer = torch.optim.Adam(model_parameters, lr=learning_rate)

      # resume model training
      if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
            print(f'{feature_extractor} trained weights loaded')
      
      model_train(start_epoch, stop_epoch, train_dataloader, val_dataloader, model, loss_fn, optimizer, device)






