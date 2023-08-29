import argparse
import glob
import numpy as np
import os

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'IDC Grading script %s' %(script))
    parser.add_argument('--feature_extractor'       , default='resnet50',      help='feature extractor: efficientnet_b0, efficientnet_v2_s, resnet50, mobilenet_v2') # some models are not available in torchvision models
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') 
    parser.add_argument('--unzip'   , action='store_true',  help='unzip the FBCG Dataset.zip if available') 
    parser.add_argument('--batch_size' , default=16, type=int,help ='Batch size for model training')
    parser.add_argument('--sn'       , default='none',      help='stain normalization: none, reinhard, macenko, vahadane, staingan, stainnet')
    parser.add_argument('--temp'       , default='Temp1',      help='reference image for reinhard, macenko, spcn and acd: Temp1, Temp2, Temp3, Temp4, Temp5') #staingan and stainnet will ignore this
    
    if script == 'train':
        parser.add_argument('--lr'   , default=0.001, type=float, help='Learning Rate')
        parser.add_argument('--save_freq'   , default=10, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=100, type=int, help ='Stopping epoch') 
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--cross_val'      , action='store_true', help='perform stratified cross validation')
        parser.add_argument('--ksplit'   , default=5, type=int, help='K-fold split')

    elif script == 'test':
      parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
    


    else:
       raise ValueError('Unknown script')
        

    return parser.parse_args()

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file
