import sys
import torch.nn as nn
import torch.optim as optim
import core.models as mdet

STE_inputs = {
    # input files
    'csv_train_full': '/home/nnamdi_20/activeSpeakersContext/dataset/csv/train/ava_activespeaker_train_augmented.csv',
    'csv_val_full': '/home/nnamdi_20/activeSpeakersContext/dataset/csv/val/ava_activespeaker_val_augmented.csv',

    # Data config
    'audio_dir': '/home/nnamdi_20/activeSpeakersContext/dataset/sliced_audio/',
    'video_dir': '/home/nnamdi_20/activeSpeakersContext/dataset/facecrops/',
    'models_out': '/home/nnamdi_20/activeSpeakersContext/dataset/ste_models_out/'
}

ASC_inputs = {
    # input files
    'features_train_full': 'STE_CSVs/STE_csvs/train/',
    'features_val_full': 'STE_CSVs/STE_csvs/val/',

    # Data config
    'models_out': 'ASC_Output/'
}

ASC_inputs_forward = {
    # input files
    'features_train_full': '...',
    'features_val_full': '.../val_forward'
}

#Optimization params
STE_optimization_params = {
    # Net Arch
    'backbone': mdet.resnet18_two_streams,

    # Optimization config
    'optimizer': optim.Adam,
    'criterion': nn.CrossEntropyLoss(),
    'learning_rate': 3e-4,
    'epochs': 100,
    'step_size': 40,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 32,
    'threads': 2
}

STE_forward_params = {
    # Net Arch
    'backbone': mdet.resnet18_two_streams_forward,

    # Batch Config
    'batch_size': 1,
    'threads': 1
}

ASC_optimization_params = {
    # Optimizer config
    'eps':3e-06, 
    'weight_decay':1e-4, 
    'amsgrad':False,


    ######
    'optimizer': optim.AdamW,
    'criterion': nn.CrossEntropyLoss(),
    'learning_rate': 3e-6,
    'epochs': 15,
    'step_size': 3,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 64,
    'threads': 0
}

ASC_forward_params = {
    # Batch Config
    'batch_size': 1,
    'threads': 1
}
