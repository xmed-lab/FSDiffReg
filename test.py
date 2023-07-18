import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from math import *
import time
import numpy as np
import torch.nn.functional as F




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/.json',
                        help='JSON file for configuration')
    parser.add_argument('-w', '--weights', type=str, default='',
                        help='weights file for validation')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    phase = 'test'
    dataset_opt=opt['datasets']['test']
    test_set = Data.create_dataset_3D(dataset_opt, phase)
    test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    print('Dataset Initialized')

    opt['path']['resume_state']=args.weights
    # model
    diffusion = Model.create_model(opt)
    print("Model Initialized")
    # Train

    registDice = np.zeros((len(test_set), 5))
    originDice = np.zeros((len(test_set), 5))
    registTime = []
    registTime = []
    print('Begin Model Evaluation.')
    idx_ = 0
    result_path = '{}'.format(opt['path']['results'])
        
    os.makedirs(result_path, exist_ok=True)
    print(len(test_loader))
    for istep,  test_data in enumerate(test_loader):
        idx_ += 1
        dataName=istep
        time1 = time.time()
        diffusion.feed_data(test_data)
        diffusion.test_registration()
        time2 = time.time()
        visuals = diffusion.get_current_registration()
        # print(visuals['contF'].shape)
        defm_frames_visual = visuals['contD'].squeeze(0).numpy().transpose(0, 2, 3, 1)
        flow_frames = visuals['contF'].numpy().transpose(0, 3, 4, 2, 1)
        flow_frames_ES = flow_frames[-1]
        sflow = torch.from_numpy(flow_frames_ES.transpose(3, 2, 0, 1).copy()).unsqueeze(0)
        sflow = Metrics.transform_grid(sflow[:, 0], sflow[:, 1], sflow[:, 2])
        nb, nc, nd, nh, nw = sflow.shape
        segflow = torch.FloatTensor(sflow.shape).zero_()
        segflow[:, 2] = (sflow[:, 0] / (nd - 1) - 0.5) * 2.0  # D[0 -> 2]
        segflow[:, 1] = (sflow[:, 1] / (nh - 1) - 0.5) * 2.0  # H[1 -> 1]
        segflow[:, 0] = (sflow[:, 2] / (nw - 1) - 0.5) * 2.0  # W[2 -> 0]
        origin_seg = test_data['MS'].squeeze()
        origin_seg = origin_seg.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        regist_seg = F.grid_sample(origin_seg.cuda().float(), (segflow.cuda().float().permute(0, 2, 3, 4, 1)),
                                   mode='nearest')
        regist_seg_=regist_seg.permute(0,1,3, 4, 2)
        regist_seg = regist_seg.squeeze().cpu().numpy().transpose(1, 2, 0)
        label_seg = test_data['FS'][0].cpu().numpy()
        origin_seg = test_data['MS'][0].cpu().numpy()
        vals_regist = Metrics.dice_ACDC(regist_seg, label_seg)[::3]
        vals_origin = Metrics.dice_ACDC(origin_seg, label_seg)[::3]
        
        registDice[istep] = vals_regist
        originDice[istep] = vals_origin
        print('---- Original Dice: %03f | Deformed Dice: %03f' % (np.mean(vals_origin), np.mean(vals_regist)))
        registTime.append(time2 - time1)
        time.sleep(1)
    omdice, osdice = np.mean(originDice), np.std(originDice)
    mdice, sdice = np.mean(registDice), np.std(registDice)
    mtime, stime = np.mean(registTime), np.std(registTime)

    print()
    print('---------------------------------------------')
    print('Total Dice and Time Metrics------------------')
    print('---------------------------------------------')
    print('origin Dice | mean = %.3f, std= %.3f' % (omdice, osdice))
    print(f'origin detailed Dice | mean = {np.mean(originDice,axis=0)}({np.std(originDice,axis=0)})')
    print('Deform Dice | mean = %.3f, std= %.3f' % (mdice, sdice))
    print(f'Deform detailed Dice | mean = {np.mean(registDice,axis=0)}({np.std(registDice,axis=0)})')
    print('Deform Time | mean = %.3f, std= %.3f' % (mtime, stime))
