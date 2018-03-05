################################################################################
##                                                                            ##
##      REHEATING & INTERPOLATION                                             ##
##     ---------------------------                                            ##
##                                                                            ##
##      This program takes a single sample and varies the durationg of        ##
##      the initial training.                                                 ##
##                                                                            ##
################################################################################
##                                                                            ##
##      TODO:                                                                 ##
##                                                                            ##
##      * Save state_dict's for a bunch of intermediate points, in order      ##
##        to do an interpolation in interpolation.py                          ##
##                                                                            ##
##      * Check whether Loss = NaN and quit()                                 ##
##                                                                            ##
##      * ResNets?                                                            ##
##                                                                            ##
################################################################################
##                                                                            ##
##      RESULTS:                                                              ##
##                                                                            ##
##      * Fixing LR or BS have, it seems, qualitatively similar effects.      ##
##        Still, it looks like Fixed-BS show cleaner results (large LR        ##
##        have difficulties converging?)                                      ##
##                                                                            ##
##      * The better the system is minimized, the rarer the jumps.            ##
##        The same is true if the network is larger.                          ##
##                                                                            ##
##      * Both seem to imply smaller barriers in the landscape.               ##
##                                                                            ##
################################################################################


import os
import pickle
import numpy as np
from collections import OrderedDict
from decimal import Decimal
import torch
from torch import Tensor, nn, optim, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sampler import RandomSampler, load_batch


# --  Models  ---------------------------------------------------------------- #


import models
from models.resnet import resnet18


# --  Datasets  -------------------------------------------------------------- #


# CIFAR-10 dataset: 3 channels, 10 classes, 32x32 pixels
# Normalized
trainset = list(datasets.CIFAR10(
	'../data/',
	train = True,
	download = True,
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
))


# --  Training function  ----------------------------------------------------- #


def train_and_save(model, trainset, lr, bs, minimization_time, file_state, file_losses, time_delay = 0, time_factor = None, losses_dump = None):
    """This function trains a model by model and saves both its state_dict at
    the end and the losses (on a log scale). It takes care of cuda().

     -- model,          the model (the user should call .cuda() before)
     -- trainset,       the dataset
     -- lr,             the learning rate
     -- bs,             the batch size
     -- min_time,       training time
     -- f_state,        where to save the state_dict
     -- f_losses,       where to save the loss evolution
     -- time_factor,    multiplicative factor for log time intervals (to save data)
     -- losses_dump     it can be passed an open() ref, data will be dumped there;
                        it is useful to train several time the same system
        )
    """

    if cuda.is_available(): model.cuda()
    model.train()  # not necessary in this simple model, but I keep it for the sake of generality
    optimizer = optim.SGD(model.parameters(), lr = lr)	# learning rate

    trainloader = DataLoader(
        trainset,								# dataset
        batch_size = bs,						# batch size
        pin_memory = cuda.is_available(),		# speed-up for gpu's
        sampler = RandomSampler(len(trainset))	# no epochs
    )

    if time_factor == None: time_factor = minimization_time**(1.0/200)

    next_t = 1.0*lr  # Times are multiplied by the LR
    batch = 0

    # NOTE: if losses_dump is an open file, use it, regardless of 'file_losses'
    no_file = bool(losses_dump == None)
    if no_file: losses_dump = open(file_losses, 'wb')

    for data, target in load_batch(trainloader, cuda = cuda.is_available()):
        batch += 1
        if batch*lr > minimization_time:  # Times are multiplied by the LR
            break

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, size_average = True)
        loss.backward()
        optimizer.step()

        # Times are multiplied by the LR
        # Save also the last step!
        if batch*lr > next_t or (batch + 1)*lr > minimization_time:
            # I want to save the average loss on the total training set
            avg_loss = 0
            total_trainloader = DataLoader(
                trainset,
                batch_size = 1024,  # I don't need small batches for this
                pin_memory = cuda.is_available(),
                sampler = RandomSampler(len(trainset))
            )

            for data, target in load_batch(total_trainloader, cuda = cuda.is_available(), only_one_epoch = True):
                output = model(data)
                avg_loss += F.nll_loss(output, target, size_average = False).data[0]

            pickle.dump(( time_delay + batch*lr, avg_loss/len(trainset) ), losses_dump)
            next_t *= time_factor

    if no_file == None: losses_dump.close()

    state_dict = model.state_dict()  # == losses[-1]['state_dict']
    torch.save(state_dict, file_state)

    return state_dict

def do_reheating_cycle(lrs, bss, trainset, preparation_times, relaxation_time, OUTPUT_DIR):
    """This function performs a cold minimization, then it reheats the system at
    some given temperatures and saves losses and states in the given files.

     -- lrs, bss,           these are the sets of LRs and BSs that specify the temperature
     -- preparation_times,  times at which reheating starts (SORTED)
     -- relaxation_time,    training time after reheating
     -- OUTPUT_DIR          directory to save the output data (it is created in case)
    """

    # create folder fot output data, if it does not exist
    if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    cold_model = resnet18()
    cold_state_dict = None
    prev_time = 0

    # The cold training will be saved all in the same file:
    cold_losses_dump = open(OUTPUT_DIR + '/cold_losses_lr={}_bs={}.p'.format(lrs[0], bss[0]), 'wb')

    # This outer cycle loops through the preparation_times, and minimizes in
    # each interval:
    print("Training the system")
    for preparation_time in preparation_times:
        cold_state_dict = train_and_save(
            cold_model, trainset, lrs[0], bss[0], preparation_time - prev_time,
            time_delay = prev_time,
            file_state = OUTPUT_DIR + '/cold_trained_time={:1.0}_lr={}_bs={}.p'.format(Decimal(preparation_time), lrs[0], bss[0]),
            file_losses = None,
            losses_dump = cold_losses_dump
        )

        prev_time = preparation_time

        # The inner cycle loops through the reheating temperatures (skip first one)
        print("First branching at t={}:".format(preparation_time))
        for lr, bs in list(zip(lrs, bss))[1:]:
            print("Heating up to T={}, lr={}, bs={}".format(lr/bs, lr, bs))
            reheated_model = resnet18()
            # In this experiment I am always starting from the same state!
            reheated_model.load_state_dict(cold_state_dict)

            train_and_save(
                reheated_model, trainset, lr, bs, relaxation_time,
                time_delay = prev_time,
                file_state = OUTPUT_DIR + '/reheated_trained_time={:1.0}_lr={}_bs={}.p'.format(Decimal(preparation_time), lr, bs),
                file_losses = OUTPUT_DIR + '/reheated_losses_time={:1.0}_lr={}_bs={}.p'.format(Decimal(preparation_time), lr, bs)
            )

    # Perform a final relaxation, in order to be compared with the reheated curves
    cold_state_dict = train_and_save(
        cold_model, trainset, lrs[0], bss[0], relaxation_time,
        time_delay = prev_time,
        file_state = OUTPUT_DIR + '/cold_trained_time={:1.0}_lr={}_bs={}.p'.format(Decimal(preparation_time), lrs[0], bss[0]),
        file_losses = None,
        losses_dump = cold_losses_dump
    )
    cold_losses_dump.close()


# ==  MAIN  ================================================================== #


# Duration of training: at each time, a bunch of reheated copies of the system
# are trained for a time relaxation_time.
# NOTE (important): times are expressed as BATCH TIME * LR!
ref_lr = 0.01
preparation_times = [ ref_lr*int(t) for t in [1e5, 5e5, 1e6] ]
relaxation_time = ref_lr*1e6


lrs = [0.03, 0.03, 0.03, 0.05, 0.03]
bss = [150, 125, 100, 100, 50]
print("Training with temps = [" + ", ".join([ "{}/{}".format(lr, bs) for lr, bs in zip(lrs, bss) ]) + "]")
do_reheating_cycle(
    lrs, bss, trainset,
    preparation_times = preparation_times,
    relaxation_time = relaxation_time,
    OUTPUT_DIR = 'reheating_same_sample_data_ResNet18-10/fixed_lr_cold_lr={}_bs={}'.format(lrs[0], bss[0])
)


lrs = [0.03, 0.03, 0.05, 0.03]
bss = [125, 100, 100, 50]
print("Training with temps = [" + ", ".join([ "{}/{}".format(lr, bs) for lr, bs in zip(lrs, bss) ]) + "]")
do_reheating_cycle(
    lrs, bss, trainset,
    preparation_times = preparation_times,
    relaxation_time = relaxation_time,
    OUTPUT_DIR = 'reheating_same_sample_data_ResNet18-10/fixed_lr_cold_lr={}_bs={}'.format(lrs[0], bss[0])
)
