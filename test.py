import torch 
import os

# import from other python files
import config
from io_utils import parse_args, get_best_file , get_assigned_file
from model import IDC_Grading_Model, test
from dataset import FBCG
from get_temp import get_temp

if __name__=='__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # initialise result dir
  result_dir = config.RECORD_DIR
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  params = parse_args('test')

  # initialise testing parsing arguments
  feature_extractor = params.feature_extractor
  train_aug = params.train_aug
  unzip = params.unzip
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
 
  test_dataloader, y_true = FBCG(data_path = config.DATA_PATH, zip_path = config.ZIP_PATH, augmentation = train_aug, unzip = unzip, batch_size=batch_size, test_mode=True, sn = params.sn, temp_dir = temp_dir)
  # initialise CNN model pass to device
  model = IDC_Grading_Model(feature_extractor).to(device)

  # initialise checkpoint directory
  checkpoint_dir = '%s/checkpoints/%s' %(config.SAVE_DIR, params.feature_extractor)

  if params.train_aug:
    checkpoint_dir += '_aug'
  # if stain normalisation is applied
  if params.sn != 'none':
      params.checkpoint_dir += f'_{params.sn}'
      params.checkpoint_dir += f'_{params.temp}'

  # load the model with best weights or from specific epochs
  if params.save_iter != -1:
    modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
  else:
    modelfile   = get_best_file(checkpoint_dir)
    
  if modelfile is not None:
      tmp = torch.load(modelfile)
      model.load_state_dict(tmp['state'])

  # model inference
  acc = test(y_true, test_dataloader, model, device)
  print(f"Model: {feature_extractor}, Test acc: {(acc):>0.2f}%")

  # save results into results.txt
  with open(os.path.join(result_dir, 'results.txt'), 'a') as f:
    aug_str = '-aug' if params.train_aug else ''
    exp_setting = f'{feature_extractor} -{params.sn} -{params.temp} {aug_str}'
    acc_str = f'Test acc: {(acc):>0.2f}%'
    f.write( 'Setting: %s, Acc: %s \n' %(exp_setting,acc_str))

  
 
