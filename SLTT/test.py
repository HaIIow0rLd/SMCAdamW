import datetime
import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from models import spiking_resnet_imagenet, spiking_resnet, spiking_vgg_bn
from modules import neuron
import argparse
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import surrogate as surrogate_sj
from modules import surrogate as surrogate_self
from utils import accuracy
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchtoolbox.transform import Cutout
from utils.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from utils.augmentation import ToPILImage, Resize, ToTensor
import collections
import random
import numpy as np
import logging
from datetime import datetime, timedelta

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.optim import Optimizer
from typing import Tuple
import matplotlib.pyplot as plt
import torch.optim as optim

from adopt import ADOPT

def main():

    # Track metrics across epochs
    
    train_accuracies = []
    test_accuracies = []
    best_epoch = 0
    max_train_accuracy = 0
    max_train_accuracy_epoch = 0
    min_train_loss = float('inf')

    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('-seed', default=2022, type=int)
    parser.add_argument('-name', default='', type=str, help='specify a name for the checkpoint and log files')
    parser.add_argument('-T', default=6, type=int, help='simulating time-steps')
    parser.add_argument('-tau', default=1.1, type=float, help='a hyperparameter for the LIF model')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default='./data', help='directory of the used dataset')
    parser.add_argument('-dataset', default='DVSCIFAR10', type=str, help='should be cifar10, cifar100, DVSCIFAR10, dvsgesture, or imagenet')
    parser.add_argument('-out_dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-surrogate', default='triangle', type=str, help='used surrogate function. should be sigmoid, rectangle, or triangle')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-pre_train', type=str, help='load a pretrained model. used for imagenet')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, help='use which optimizer. SGD or Adam or AdamW or SMCAdamW or SMCAdam or RMSprop or SMCRMSprop or NAdam or SMCNAdam or Adagrad or SMCAdagrad', default='AdamW')
    parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=100, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=300, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-model', type=str, default='spiking_vgg11_bn', help='use which SNN model')
    parser.add_argument('-drop_rate', type=float, default=0.0, help='dropout rate. used for DVSCIFAR10 and dvsgesture')
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-loss_lambda', type=float, default=0.05, help='the scaling factor for the MSE term in the loss')
    parser.add_argument('-mse_n_reg', action='store_true', help='loss function setting')
    parser.add_argument('-loss_means', type=float, default=1.0, help='used in the loss function when mse_n_reg=False')
    parser.add_argument('-K', default=2, type=int, help='the number of trained time steps')

    parser.add_argument('-lambda_sm', default=0.1, type=float, help='sliding mode gradient weight')
    parser.add_argument('-alpha_sm', default=1e-3, type=float, help='sliding mode correction strength')
    parser.add_argument('-k', default=5.0, type=float, help='switching steepness')


    args = parser.parse_args()
    print(args)

    _seed_ = args.seed
    random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    np.random.seed(_seed_)

    # ==================== 1. Logging setup ====================
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f'Created output directory: {args.out_dir}')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_info = f"{args.dataset}_{args.model}"
    log_filename = os.path.join(args.out_dir, f"{timestamp}_{model_info}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="INFO:root:%(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("=== Training started at %s ===", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Log file: %s", log_filename)
    logging.info("Model: %s, Dataset: %s", args.model, args.dataset)
    logging.info("Command line arguments: %s", args)

    if torch.cuda.is_available():
        logging.info("Using GPU device %d", torch.cuda.current_device())
    logging.info("Loading dataset...")


    # ==================== 2. Data ====================
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        c_in = 3
        if args.dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            num_classes = 10
            normalization_mean = (0.4914, 0.4822, 0.4465)
            normalization_std = (0.2023, 0.1994, 0.2010)
        elif args.dataset == 'cifar100':
            dataloader = datasets.CIFAR100
            num_classes = 100
            normalization_mean = (0.5071, 0.4867, 0.4408)
            normalization_std = (0.2675, 0.2565, 0.2761)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            Cutout(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std),
        ])

        trainset = dataloader(root=args.data_dir, train=True, download=True, transform=transform_train)
        train_data_loader = data.DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.j)

        testset = dataloader(root=args.data_dir, train=False, download=False, transform=transform_test)
        test_data_loader = data.DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j)

    elif args.dataset == 'DVSCIFAR10':
        c_in = 2
        num_classes = 10

        transform_train = transforms.Compose([
            ToPILImage(),
            Resize(48),
            ToTensor(),
        ])

        transform_test = transforms.Compose([
            ToPILImage(),
            Resize(48),
            ToTensor(),
        ])

        trainset = CIFAR10DVS(args.data_dir, train=True, use_frame=True, frames_num=args.T, split_by='number', normalization=None, transform=transform_train)
        train_data_loader = data.DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.j)

        testset = CIFAR10DVS(args.data_dir, train=False, use_frame=True, frames_num=args.T, split_by='number', normalization=None, transform=transform_test)
        test_data_loader = data.DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j)


    elif args.dataset == 'dvsgesture':
        c_in = 2
        num_classes = 11

        trainset = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
        train_data_loader = data.DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.j, drop_last=True, pin_memory=True)

        testset = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
        test_data_loader = data.DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j, drop_last=False, pin_memory=True)

    elif args.dataset == 'imagenet':
        num_classes = 1000
        traindir = os.path.join(args.data_dir, 'train')
        valdir = os.path.join(args.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.b, shuffle=True,
            num_workers=args.j, pin_memory=True)

        test_data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.b, shuffle=False,
            num_workers=args.j, pin_memory=True)

    else:
        raise NotImplementedError
    
    logging.info("Dataset %s loaded successfully", args.dataset)

    # ==================== 3. Model ====================
    if args.surrogate == 'sigmoid':
        surrogate_function = surrogate_sj.Sigmoid()
    elif args.surrogate == 'rectangle':
        surrogate_function = surrogate_self.Rectangle()
    elif args.surrogate == 'triangle':
        surrogate_function = surrogate_sj.PiecewiseQuadratic()

    neuron_model = neuron.SLTTNeuron

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        net = spiking_resnet.__dict__[args.model](neuron=neuron_model, num_classes=num_classes, neuron_dropout=args.drop_rate,
                                                tau=args.tau, surrogate_function=surrogate_function, c_in=c_in, fc_hw=1)
        logging.info('Using Resnet model')
    elif args.dataset == 'imagenet':
        net = spiking_resnet_imagenet.__dict__[args.model](neuron=neuron_model, num_classes=num_classes, neuron_dropout=args.drop_rate,
                                                        tau=args.tau, surrogate_function=surrogate_function, c_in=3)
        logging.info('Using NF-Resnet model')
    elif args.dataset == 'DVSCIFAR10' or args.dataset == 'dvsgesture':
        net = spiking_vgg_bn.__dict__[args.model](neuron=neuron_model, num_classes=num_classes, neuron_dropout=args.drop_rate,
                                                tau=args.tau, surrogate_function=surrogate_function, c_in=c_in, fc_hw=1)
        logging.info('Using VGG model')
    else:
        raise NotImplementedError
        
    total_params = sum(p.numel() for p in net.parameters())
    logging.info('Total Parameters: %.2fM', total_params / 1000000.0)
    net.cuda()
    logging.info("\n=== Network Architecture ===")
    logging.info("Model: %s", args.model)
    logging.info("Total Parameters: %.2fM", total_params / 1000000.0)

    # ==================== 4. Optimizer & scheduler ====================
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay = 0)
    elif args.opt == 'RMSprop':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99)
    elif args.opt == 'NAdam':
        optimizer = torch.optim.NAdam(net.parameters(), lr=args.lr, weight_decay = 0)
    elif args.opt == 'Adagrad':
        new_lr_for_adagrad = args.lr * 10
        optimizer = torch.optim.Adagrad(net.parameters(), lr=new_lr_for_adagrad, weight_decay=0)
    else:
        raise NotImplementedError(args.opt)
    
    # Override with ADOPT (decoupled weight decay like AdamW)
    optimizer = ADOPT(
        net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        decouple=True,
    )
      
        
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
        
    logging.info("\n=== Training Configuration ===")
    logging.info("Batch Size: %d", args.b)
    logging.info("Optimizer: %s", optimizer.__class__.__name__)
    logging.info("Optimizer Parameters:")
    for param_name in optimizer.defaults:
        logging.info("  - %s: %s", param_name, optimizer.defaults[param_name])
    logging.info("Starting training with %d epochs", args.epochs)

    # ==================== 5. Resume / preload ====================
    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        logging.info('Resuming from checkpoint: %s', args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        logging.info('Start epoch: %d, max test acc: %.4f', start_epoch, max_test_acc)

    if args.pre_train:
        logging.info('Loading pre-trained model: %s', args.pre_train)
        checkpoint = torch.load(args.pre_train, map_location='cpu')
        state_dict2 = collections.OrderedDict([(k, v) for k, v in checkpoint['net'].items()])
        net.load_state_dict(state_dict2)
        logging.info('Using pre-trained model, max test acc: %.4f', checkpoint['max_test_acc'])

    # ==================== 6. Train / test ====================
    criterion_mse = nn.MSELoss()
    
    total_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        ############### training ###############
        epoch_start_time = time.time()
        net.train()

        train_loss = 0
        train_acc = 0
        train_samples = 0
        batch_idx = 0

        logging.info('\nEpoch: %d', epoch)

        for frame, label in train_data_loader:
            batch_idx += 1
            if args.dataset != 'DVSCIFAR10':
                frame = frame.float().cuda()
                if args.dataset == 'dvsgesture':
                    frame = frame.transpose(0, 1)
            t_step = args.T

            train_step = torch.randperm(args.T)[:args.K]

            label = label.cuda()

            batch_loss = 0
            optimizer.zero_grad()
            for t in range(t_step):
                if args.dataset == 'DVSCIFAR10':
                    input_frame = frame[t].float().cuda()
                elif args.dataset == 'dvsgesture':
                    input_frame = frame[t]
                else:
                    input_frame = frame
                if args.amp:
                    if t in train_step:
                        with amp.autocast():
                            if t == 0:
                                out_fr = net(input_frame)
                                total_fr = out_fr.clone().detach()
                            else:
                                out_fr = net(input_frame)
                                total_fr += out_fr.clone().detach()
                            # Calculate the loss
                            if args.loss_lambda > 0.0:  # the loss is a cross entropy term plus a mse term
                                if args.mse_n_reg:  # the mse term is not treated as a regularizer
                                    label_one_hot = F.one_hot(label, num_classes).float()
                                else:
                                    label_one_hot = torch.zeros_like(out_fr).fill_(args.loss_means).to(out_fr.device)
                                mse_loss = criterion_mse(out_fr, label_one_hot)
                                loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                            else:  # the loss is just a cross entropy term
                                loss = F.cross_entropy(out_fr, label) / t_step
                        scaler.scale(loss).backward()
                        batch_loss += loss.item()
                        train_loss += loss.item() * label.numel()
                    else:
                        with amp.autocast():
                            with torch.no_grad():
                                if t == 0:
                                    out_fr = net(input_frame)
                                    total_fr = out_fr.clone().detach()
                                else:
                                    out_fr = net(input_frame)
                                    total_fr += out_fr.clone().detach()

                else:
                    raise NotImplementedError('Please use amp.')

            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            train_samples += label.numel()
            train_acc += (total_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_loss /= train_samples
        train_acc /= train_samples
        
        train_accuracies.append(train_acc)
        
        if train_acc > max_train_accuracy:
            max_train_accuracy = train_acc
            max_train_accuracy_epoch = epoch
        if train_loss < min_train_loss:
            min_train_loss = train_loss

        lr_scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        logging.info("Train Accuracy: %.3f (Max: %.3f at epoch %d). Loss: %.3f (Min: %.3f). Time: %.2f seconds", 
                    100. * train_acc, 100 * max_train_accuracy, max_train_accuracy_epoch, train_loss, min_train_loss, epoch_time)

        ############### testing ###############
        test_start_time = time.time()
        net.eval()

        test_loss = 0
        test_acc = 0
        test_samples = 0
        batch_idx = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                batch_idx += 1
                if args.dataset != 'DVSCIFAR10':
                    frame = frame.float().cuda()
                    if args.dataset == 'dvsgesture':
                        frame = frame.transpose(0, 1)
                label = label.cuda()
                t_step = args.T
                total_loss = 0

                for t in range(t_step):
                    if args.dataset == 'DVSCIFAR10':
                        input_frame = frame[t].float().cuda()
                    elif args.dataset == 'dvsgesture':
                        input_frame = frame[t]
                    else:
                        input_frame = frame
                    if t == 0:
                        out_fr = net(input_frame)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = net(input_frame)
                        total_fr += out_fr.clone().detach()
                    # Calculate the loss
                    if args.loss_lambda > 0.0: # the loss is a cross entropy term plus a mse term
                        if args.mse_n_reg:  # the mse term is not treated as a regularizer
                            label_one_hot = F.one_hot(label, num_classes).float()
                        else:
                            label_one_hot = torch.zeros_like(out_fr).fill_(args.loss_means).to(out_fr.device)
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                    else: # the loss is just a cross entropy term
                        loss = F.cross_entropy(out_fr, label) / t_step
                    total_loss += loss

                test_samples += label.numel()
                test_loss += total_loss.item() * label.numel()
                test_acc += (total_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_loss /= test_samples
        # test_acc = (test_acc+4)/ test_samples
        test_acc = (test_acc)/ test_samples

        test_accuracies.append(test_acc)

        test_time = time.time() - test_start_time

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            best_epoch = epoch
            
        logging.info("Test Accuracy: %.3f (Best: %.3f at epoch %d). Time: %.2f seconds", 
                    100. * test_acc, 100 * max_test_acc, best_epoch, test_time)

        logging.info("after one epoch: %.5fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))


        if epoch == args.epochs - 1:
            total_training_time = time.time() - total_start_time
            
            avg_train_acc = sum(train_accuracies) / len(train_accuracies)
            avg_test_acc = sum(test_accuracies) / len(test_accuracies)
            
            logging.info("\n=== Training Summary ===")
            logging.info("Total training time: %.5f seconds", total_training_time)
            logging.info("Best Testing Accuracy: %.5f at epoch: %d", 100 * max_test_acc, best_epoch)
            logging.info("Best Training Accuracy: %.5f at epoch: %d", 100 * max_train_accuracy, max_train_accuracy_epoch)
            logging.info("Batch Size: %d", args.b)
            logging.info("Optimizer: %s with parameters:", optimizer.__class__.__name__)
            for param_name in optimizer.defaults:
                if isinstance(optimizer.defaults[param_name], (int, float)):
                    logging.info("  - %s: %.5f", param_name, optimizer.defaults[param_name])
                else:
                    logging.info("  - %s: %s", param_name, optimizer.defaults[param_name])
            logging.info("Average Training Accuracy: %.5f", 100 * avg_train_acc)
            logging.info("Average Testing Accuracy: %.5f", 100 * avg_test_acc)
            logging.info("=== Training finished at %s ===", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
            plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
            
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Testing Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.ylim(0, 1)
            
            plt.xlim(1, args.epochs)
            
            log_file = logging.getLoggerClass().root.handlers[0].baseFilename
            img_file = os.path.splitext(log_file)[0] + ".png"
            plt.savefig(img_file, dpi=300)
            plt.close()
if __name__ == '__main__':
    main()
