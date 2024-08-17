import sys
import os
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from core.dataset import ASCFeaturesDataset
from core.optimization import optimize_asc
import core.models as mdet

import core.config as exp_conf
from core.io import set_up_log_and_ws_out

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
     
    #experiment Reproducibility
    torch.manual_seed(11)
    torch.cuda.manual_seed(22)
    torch.backends.cudnn.deterministic = True

    time_lenght = int(sys.argv[1])
    time_stride = int(sys.argv[2])
    speakers = int(sys.argv[3])
    cuda_device_number = str(sys.argv[4])

    model_name = 'ASC'+'_len_' + str(time_lenght) + '_stride_'+str(time_stride) + '_speakers_'+str(speakers)
    io_config = exp_conf.ASC_inputs
    opt_config = exp_conf.ASC_optimization_params

    # io config
    log, target_models = set_up_log_and_ws_out(io_config['models_out'],
                                               opt_config, model_name)

    # cuda config
    model = mdet.ASC_Net(clip_number=time_lenght, candidate_speakers=speakers)
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    print(device)
    model = nn.DataParallel(model)
    model.to(device)

    criterion = opt_config['criterion']
    optimizer = opt_config['optimizer'](model.parameters(),
                                        lr=opt_config['learning_rate'])
    
                                         
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt_config['step_size'],
                                    gamma=opt_config['gamma'])

    dataset_train = ASCFeaturesDataset(io_config['features_train_full'],
                                       time_lenght=time_lenght,
                                       time_stride=time_stride,
                                       candidate_speakers=speakers)
    dataset_val = ASCFeaturesDataset(io_config['features_val_full'],
                                     time_lenght=time_lenght,
                                     time_stride=time_stride,
                                     candidate_speakers=speakers)

    dl_train = DataLoader(dataset_train, batch_size=opt_config['batch_size'],
                          shuffle=True, num_workers=opt_config['threads'])
    dl_val = DataLoader(dataset_val, batch_size=opt_config['batch_size'],
                        shuffle=False, num_workers=opt_config['threads'])

    optimize_asc(model, dl_train, dl_val, device, criterion, optimizer,
                 scheduler, num_epochs=opt_config['epochs'],
                 models_out=target_models, log=log)
