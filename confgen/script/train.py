#coding: utf-8

import os
import sys
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print('project path is {}'.format(project_path))
sys.path.append(project_path)
import argparse
import numpy as np
import random
import pickle
import yaml
from easydict import EasyDict

import torch
from confgf import models, dataset, runner, utils
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='clofnet')
    parser.add_argument('--config_path', type=str, help='path of dataset', required=True)
    parser.add_argument('--seed', type=int, default=2021, help='overwrite config seed')

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.seed != 2021:
        config.train.seed = args.seed

    if config.train.save and config.train.save_path is not None:
        config.train.save_path = os.path.join(config.train.save_path, config.train.Name)
        if not os.path.exists(config.train.save_path):
            os.makedirs(config.train.save_path)


    # check device
    gpus = list(filter(lambda x: x is not None, config.train.gpus))
    assert torch.cuda.device_count() >= len(gpus), 'do you set the gpus in config correctly?'
    device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')
    print("Let's use", len(gpus), "GPUs!")
    print("Using device %s as main device" % device)    
    config.train.device = device
    config.train.gpus = gpus

    print(config)

    # set random seed
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)        
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.benchmark = True
    print('set seed for random, numpy and torch')


    load_path = os.path.join(config.data.base_path, '%s_processed' % config.data.dataset)
    print('loading data from %s' % load_path)

    train_data = []
    val_data = []
    test_data = []

    if config.data.train_set is not None:          
        with open(os.path.join(load_path, config.data.train_set), "rb") as fin:
            train_data = pickle.load(fin)
    if config.data.val_set is not None:
        with open(os.path.join(load_path, config.data.val_set), "rb") as fin:
            val_data = pickle.load(fin)
    print(val_data[0])
    #torch.save(val_data[0], 'val.pt')
    #sys.exit("saved val object object successful!")
    print('train size : %d  ||  val size: %d  ||  test size: %d ' % (len(train_data), len(val_data), len(test_data)))
    print('loading data done!')
    
    batch_size = config.train.batch_size 
    train_batch_size, val_batch_size, test_batch_size = batch_size, batch_size, batch_size
    
    train_data, val_data, test_data = (
        dataset.batched_QM9(data=train_data, batch_size=train_batch_size),
        dataset.batched_QM9(data=val_data, batch_size=val_batch_size),
        dataset.batched_QM9(data=test_data, batch_size=test_batch_size),
    )
    
    # get dataloaders, note that batch size must be 1 because batch is already divided in dataset.
    #train_data, val_data, test_data = (
    #    DataLoader(train_dataset, num_workers=4, batch_size=1, persistent_workers=False, shuffle=True), 
    #    DataLoader(val_dataset, num_workers=4, batch_size=1, persistent_workers=False, shuffle=False), 
    #    DataLoader(test_dataset, num_workers=4, batch_size=1, persistent_workers=False, shuffle=False),
    #)
    #original ClofNet data:
    #transform = None

    #train_data = dataset.GEOMDataset(data=train_data, transform=transform)
    #val_data = dataset.GEOMDataset(data=val_data, transform=transform)

    #test_data = dataset.GEOMDataset_PackedConf(data=test_data, transform=transform)
    
    model = models.EquiDistanceScoreMatch(config)
    optimizer = utils.get_optimizer(config.train.optimizer, model)
    scheduler = utils.get_scheduler(config.train.scheduler, optimizer)

    solver = runner.EquiRunner(train_data, val_data, test_data, model, optimizer, scheduler, gpus, config)
    if config.train.resume_train:
        solver.load(config.train.resume_checkpoint, epoch=config.train.resume_epoch, load_optimizer=True, load_scheduler=True)
    solver.train()


