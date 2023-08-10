import json
import os
import pickle
import argparse
import platform
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from Datapreprocessor import Datapreprocessor, LinearsDataset
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


# torchrun --nproc_per_node=2 --nnodes=1 --master-port 2333 informermain.py -GDMC 0,1 -e 3 -o 24 -b 90 --fixed_seed 3407 -m fedformer -d ettm1 -s 10

def check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
        "\n")
    # print(devices_info)
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def omem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.FloatTensor(256, 1024, block_mem).to(cuda_device)
    del x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-B', '--best_model', action='store_true')
    parser.add_argument('-C', '--CUDA_VISIBLE_DEVICES', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('-d', '--dataset', type=str, default='wht',
                        help='wht, gweather, etth1, etth2, ettm1, ettm2, exchange, ill, traffic(too large), finance')
    parser.add_argument('-D', '--delete_model_dic', action='store_true')
    parser.add_argument('-e', '--total_eopchs', type=int, default=20)
    parser.add_argument('-E', '--d_model', type=int, default=128)
    parser.add_argument('-f', '--fixed_seed', type=int, default=None)
    parser.add_argument('-F', '-freq', type=str, default='h', help='time encoding type')
    parser.add_argument('-G', '--gpu', action='store_true')
    parser.add_argument('-I', '--individual', action='store_true')
    parser.add_argument('-i', '--input_len', type=int, default=96)
    parser.add_argument('-k', '--kernel_size', type=int, default=25)
    parser.add_argument('-l', '--lr', type=float, default=.001)
    parser.add_argument('-m', '--model', type=str, default='mtgnn')
    parser.add_argument('-M', '--multi_GPU', action='store_true')
    parser.add_argument('-n', '--n_heads', type=int, default=8)
    parser.add_argument('-o', '--output_len', type=int, default=96)
    parser.add_argument('-O', '--occupy_mem', action='store_true')
    parser.add_argument('-s', '--stride', type=int, default=1)
    parser.add_argument('-S', '--save_dir', type=str, default='save/tmp')
    parser.add_argument('-t', '--train_ratio', type=float, default=.6)
    parser.add_argument('-T', '--test_set_to_validate', action='store_true')
    parser.add_argument('-v', '--valid_ratio', type=float, default=.2)
    parser.add_argument('--simple_map', action='store_true')
    parser.add_argument('--fudan', action='store_true')
    parser.add_argument('--titan', action='store_true')
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--ori_v', action='store_true')
    parser.add_argument('--random', action='store_true')
    # placeholders
    parser.add_argument('--complex_map', action='store_true')
    parser.add_argument('--not_ind', action='store_true')
    parser.add_argument('--sequential', action='store_true')
    args = parser.parse_args()
    arg_dict = vars(args)

    dataset_name = args.dataset
    input_len = args.input_len
    output_len = args.output_len
    stride = args.stride
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    batch_size = args.batch_size
    total_eopchs = args.total_eopchs
    gpu = args.gpu
    individual = args.individual
    lr = args.lr
    which_model = args.model
    save_dir = args.save_dir
    fixed_seed = args.fixed_seed
    best_model = args.best_model
    kernel_size = args.kernel_size
    delete_model_dic = args.delete_model_dic and args.best_model
    multiGPU = args.multi_GPU
    occupy_mem = args.occupy_mem
    d_model = args.d_model
    n_heads = args.n_heads
    test_set_to_validate = args.test_set_to_validate
    draw = args.draw
    simple_map = args.simple_map
    random_samples = args.random

    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    if multiGPU:
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda:0' if gpu else 'cpu')

    if occupy_mem:
        omem(local_rank)

    if (multiGPU and local_rank == 0) or not multiGPU:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(json.dumps(arg_dict, ensure_ascii=False))

    if (fixed_seed is not None) or multiGPU:
        seed = fixed_seed if fixed_seed is not None else 2333
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if multiGPU:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")

    if platform.system() == 'Windows':
        data_root = 'E:\\forecastdataset\\pkl'
        map_root = 'E:\\forecastdataset\\map'
    else:
        if args.fudan:
            data_root = '/remote-home/liuwenbo/pycproj/dataset'
            map_root = '/remote-home/liuwenbo/pycproj/forecastdata/map/'
        elif args.titan:
            data_root = '/data/user19302427/pycharmdir/forecast.dataset/pkl'
            map_root = '/data/user19302427/pycharmdir/forecast.dataset/map'
        else:
            data_root = '/home/icpc/pycharmproj/forecast.dataset/pkl/'
            map_root = '/home/icpc/pycharmproj/forecast.dataset/map/'

    data_preprocessor = None
    dataset = None
    graph = None
    if dataset_name == 'wht':
        dataset = pickle.load(open(os.path.join(data_root, 'wht.pkl'), 'rb'))
        if simple_map:
            graph = pickle.load(open(os.path.join(map_root, 'WTH.simple.pkl'), 'rb'))
        else:
            graph = pickle.load(open(os.path.join(map_root, 'WTH.complete.pkl'), 'rb'))
    elif dataset_name == 'gweather':
        dataset = pickle.load(open(os.path.join(data_root, 'weather.pkl'), 'rb'))
        if simple_map:
            graph = pickle.load(open(os.path.join(map_root, 'weather.simple.pkl'), 'rb'))
        else:
            graph = pickle.load(open(os.path.join(map_root, 'weather.complete.pkl'), 'rb'))
    elif dataset_name == 'etth1':
        dataset = pickle.load(open(os.path.join(data_root, 'ETTh1.pkl'), 'rb'))
        if simple_map:
            graph = pickle.load(open(os.path.join(map_root, 'ETTh1.simple.pkl'), 'rb'))
        else:
            graph = pickle.load(open(os.path.join(map_root, 'ETTh1.complete.pkl'), 'rb'))
    elif dataset_name == 'etth2':
        dataset = pickle.load(open(os.path.join(data_root, 'ETTh2.pkl'), 'rb'))
        if simple_map:
            graph = pickle.load(open(os.path.join(map_root, 'ETTh2.simple.pkl'), 'rb'))
        else:
            graph = pickle.load(open(os.path.join(map_root, 'ETTh2.complete.pkl'), 'rb'))
    elif dataset_name == 'ettm1':
        dataset = pickle.load(open(os.path.join(data_root, 'ETTm1.pkl'), 'rb'))
        if simple_map:
            graph = pickle.load(open(os.path.join(map_root, 'ETTm1.simple.pkl'), 'rb'))
        else:
            graph = pickle.load(open(os.path.join(map_root, 'ETTm1.complete.pkl'), 'rb'))
    elif dataset_name == 'ettm2':
        dataset = pickle.load(open(os.path.join(data_root, 'ETTm2.pkl'), 'rb'))
        if simple_map:
            graph = pickle.load(open(os.path.join(map_root, 'ETTm2.simple.pkl'), 'rb'))
        else:
            graph = pickle.load(open(os.path.join(map_root, 'ETTm2.complete.pkl'), 'rb'))
    elif dataset_name == 'exchange':
        dataset = pickle.load(open(os.path.join(data_root, 'exchange_rate.pkl'), 'rb'))
        if simple_map:
            graph = pickle.load(open(os.path.join(map_root, 'exchange_rate.simple.pkl'), 'rb'))
        else:
            graph = pickle.load(open(os.path.join(map_root, 'exchange_rate.complete.pkl'), 'rb'))
    elif dataset_name == 'ill':
        dataset = pickle.load(open(os.path.join(data_root, 'national_illness.pkl'), 'rb'))
        if simple_map:
            graph = pickle.load(open(os.path.join(map_root, 'national_illness.simple.pkl'), 'rb'))
        else:
            graph = pickle.load(open(os.path.join(map_root, 'national_illness.complete.pkl'), 'rb'))
    elif dataset_name == 'finance':
        dataset = pickle.load(open(os.path.join(data_root, 'finance.pkl'), 'rb'))
        if simple_map:
            graph = pickle.load(open(os.path.join(map_root, 'finance.simple.pkl'), 'rb'))
        else:
            graph = pickle.load(open(os.path.join(map_root, 'finance.complete.pkl'), 'rb'))
    elif dataset_name == 'traffic':
        dataset = pickle.load(open(os.path.join(data_root, 'traffic.pkl'), 'rb'))
        graph = pickle.load(open(os.path.join(map_root, 'traffic.map.pkl'), 'rb'))
    else:
        print('\033[32mno such dataset\033[0m')
        exit()
    graph = torch.Tensor(graph).to(device)
    graph = graph.transpose(0, 1) + torch.eye(graph.shape[0]).to(graph.device)
    graph = torch.where(graph == 0., -1e9, 0.)
    if (multiGPU and local_rank == 0) or not multiGPU:
        print('\033[32m', dataset.shape, '\033[0m')
    data_preprocessor = Datapreprocessor(dataset, input_len, output_len, stride=stride, random=random_samples)
    num_sensors = data_preprocessor.num_sensors

    train_input, train_gt, train_encoding = data_preprocessor.load_train_samples(encoding=True)
    valid_input, valid_gt, valid_encoding = data_preprocessor.load_validate_samples(encoding=True)
    test_input, test_gt, test_encoding = data_preprocessor.load_test_samples(encoding=True)
    train_set = LinearsDataset(train_input, train_gt)
    train_loader = DataLoader(train_set, sampler=DistributedSampler(train_set) if multiGPU else None,
                              batch_size=batch_size, shuffle=False if multiGPU else True)
    valid_loader = None
    # test_loader = None

    if (multiGPU and local_rank == 0) or not multiGPU:
        if test_set_to_validate:
            valid_loader = DataLoader(LinearsDataset(test_input, test_gt), batch_size=batch_size,
                                      shuffle=False)
        else:
            valid_loader = DataLoader(LinearsDataset(valid_input, valid_gt), batch_size=batch_size,
                                      shuffle=False)
        test_loader = DataLoader(LinearsDataset(test_input, test_gt), batch_size=batch_size,
                                 shuffle=False)

    from net import gtnet

    model = gtnet(True, True, 2, num_sensors, device, dropout=0.3, subgraph_size=5,
                  node_dim=40, dilation_exponential=2,
                  conv_channels=16, residual_channels=16,
                  skip_channels=32, end_channels=64,
                  seq_length=input_len, in_dim=1, out_dim=output_len,
                  layers=5, propalpha=0.05, tanhalpha=3, layer_norm_affline=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    pbar_epoch = None
    if (multiGPU and local_rank == 0) or not multiGPU:
        pbar_epoch = tqdm(total=total_eopchs, ascii=True, dynamic_ncols=True)
    minium_loss = 100000
    validate_loss_list = []
    last_save_step = -1
    for epoch in range(total_eopchs):
        # train
        model.train()
        graph = torch.arange(num_sensors).to(device)
        total_iters = len(train_loader)
        pbar_iter = None
        if (multiGPU and local_rank == 0) or not multiGPU:
            pbar_iter = tqdm(total=total_iters, ascii=True, dynamic_ncols=True, leave=False)
        for i, (input_x, ground_truth) in enumerate(train_loader):
            input_x = torch.unsqueeze(input_x, dim=1).transpose(2, 3)
            optimizer.zero_grad()
            input_x = input_x.to(device)
            ground_truth = ground_truth.to(device)
            output = torch.squeeze(model(input_x, graph))
            # print(output.shape,ground_truth.shape)
            loss = loss_fn(output, ground_truth)
            loss.backward()
            optimizer.step()
            if (multiGPU and local_rank == 0) or not multiGPU:
                pbar_iter.set_postfix_str('loss:{:.4f}'.format(loss.item()))
                pbar_iter.update(1)
        if (multiGPU and local_rank == 0) or not multiGPU:
            pbar_iter.close()

        # validate
        if best_model and ((multiGPU and local_rank == 0) or not multiGPU):
            model.eval()
            output_list = []
            gt_list = []
            pbar_iter = tqdm(total=len(valid_loader), ascii=True, dynamic_ncols=True, leave=False)
            pbar_iter.set_description_str('validating')
            with torch.no_grad():
                for i, (input_x, ground_truth) in enumerate(valid_loader):
                    input_x = torch.unsqueeze(input_x, dim=1).transpose(2, 3)
                    input_x = input_x.to(device)
                    output = torch.squeeze(model(input_x, graph))
                    output_list.append(output.cpu())
                    gt_list.append(ground_truth)
                    pbar_iter.update()
            pbar_iter.close()
            if torch.__version__ > '1.13.0':
                output_list = torch.concatenate(output_list, dim=0)
                gt_list = torch.concatenate(gt_list, dim=0)
            else:
                output_list = torch.cat(output_list, dim=0)
                gt_list = torch.cat(gt_list, dim=0)
            validate_loss = loss_fn(output_list, gt_list).item()
            validate_loss_list.append(validate_loss)
            pbar_epoch.set_postfix_str('eval loss:{:.4f}'.format(validate_loss))
            if validate_loss < minium_loss:
                last_save_step = epoch
                minium_loss = validate_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                pbar_epoch.set_description_str('saved at epoch %d %.4f' % (epoch + 1, minium_loss))
        if (multiGPU and local_rank == 0) or not multiGPU:
            pbar_epoch.update(1)
        if multiGPU:
            dist.barrier()
    if (multiGPU and local_rank == 0) or not multiGPU:
        pbar_epoch.close()

    # test
    if (multiGPU and local_rank == 0) or not multiGPU:
        pbar_iter = tqdm(total=len(test_loader), ascii=True, dynamic_ncols=True)
        pbar_iter.set_description_str('testing')
        output_list = []
        gt_list = []
        if best_model:
            model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
        model.eval()
        with torch.no_grad():
            for i, (input_x, ground_truth) in enumerate(test_loader):
                input_x = torch.unsqueeze(input_x, dim=1).transpose(2, 3)
                input_x = input_x.to(device)
                output = torch.squeeze(model(input_x, graph))
                if len(output.shape) < 3:
                    output = torch.unsqueeze(output, dim=0)
                output_list.append(output.cpu())
                gt_list.append(ground_truth)
                pbar_iter.update(1)
        pbar_iter.close()
        if torch.__version__ > '1.13.0':
            output_list = torch.concatenate(output_list, dim=0)
            gt_list = torch.concatenate(gt_list, dim=0)
        else:
            output_list = torch.cat(output_list, dim=0)
            gt_list = torch.cat(gt_list, dim=0)
        test_loss = loss_fn(output_list, gt_list).item()
        mae_loss = torch.mean(torch.abs(output_list - gt_list)).item()
        print('\033[32mmse loss:{:.4f} mae loss:{:.4f}\033[0m'.format(test_loss, mae_loss))
        result_dict = arg_dict
        result_dict['mse'] = test_loss
        result_dict['mae'] = mae_loss
        result_dict['save_step'] = last_save_step
        print(json.dumps(result_dict, ensure_ascii=False), file=open(os.path.join(save_dir, 'result.json'), 'w'))
        if delete_model_dic:
            os.remove(os.path.join(save_dir, 'best_model.pth'))
            print('\033[33mdeleted model.pth\033[0m')
        if multiGPU:
            dist.destroy_process_group()
        if draw:
            plt.figure()
            plt.plot(validate_loss_list)
            plt.savefig(os.path.join(save_dir, 'validate_loss.png'))
            plt.close()
