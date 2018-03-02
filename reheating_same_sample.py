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
##      * check whether Loss = NaN and quit()                                 ##
##                                                                            ##
##      * start from larger temperatures                                      ##
##                                                                            ##
################################################################################
##                                                                            ##
##      RESULTS:                                                              ##
##                                                                            ##
##      * It seems that the better the system is prepared (i.e. the           ##
##        longer the cold training), the rarer are the jumps.                 ##
##                                                                            ##
##        It is not clear why this happens.                                   ##
##                                                                            ##
##      * Fixing LR or BS have, it seems, qualitatively similar effects.      ##
##        Still, it looks like Fixed-BS show cleaner results (large LR        ##
##        have difficulties converging?)                                      ##
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


# --  Models  ---------------------------------------------------------------- #


def custom_conv2d(conv2d):
    def decorated(input_features, output_features, kernel_size, stride = 1, padding = 0, *args, **kwargs):
        def convert(image_size):
            return (image_size + 2*padding - kernel_size)//stride + 1
        return conv2d(input_features, output_features, kernel_size, stride, padding, *args, **kwargs), convert
    return decorated

conv2d = custom_conv2d(torch.nn.Conv2d)

class SimpleNet(torch.nn.Module):
    """Simple convolutional networ: 2 conv layers followed by 2 fc layers.

     model = SimpleNet(# input channels, # num of output classes, image_size)
     model(data) performs the forward computation
    """

    def __init__(self, input_features, output_classes, image_size):
        super(SimpleNet, self).__init__()
        self.conv0, f_size = conv2d(input_features, 20, kernel_size = 5, stride = 1, padding = 5 // 2)
        image_size = f_size(image_size)
        self.conv1, f_size = conv2d(20, 20, kernel_size = 5, stride = 2, padding = 0)
        image_size = f_size(image_size)
        self.conv2, f_size = conv2d(20, 20, kernel_size = 5, stride = 2, padding = 0)
        image_size = f_size(image_size)
        self.fc1 = torch.nn.Linear(20*image_size**2, 60)
        self.fc2 = torch.nn.Linear(60, output_classes)

    def forward(self, x):
        x = F.relu( self.conv0(x) )
        x = F.relu( self.conv1(x) )
        x = F.relu( self.conv2(x) )
        x = x.view(x.size(0), -1)
        x = F.relu( self.fc1(x) )
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)
        return x


# --  Datasets  -------------------------------------------------------------- #


# Fashion-MNIST dataset: 1 channel, 10 classes, 28x28 pixels
# Normalized as MNIST -- I should probably change it
trainset = list(datasets.FashionMNIST(
	'../data/',
	train = True,
	download = True,
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
))

#testset = list(datasets.FashionMNIST('../data/', train = False, download = True, transform = transforms.Compose([ \
#    transforms.ToTensor(),
#    transforms.Normalize((0.1307,), (0.3081,))
#])))


# --  Other definitions  ----------------------------------------------------- #


class RandomSampler:
    """RandomSampler is a sampler for torch.utils.data.DataLoader.

    Each batch is independent (i.e. with repetition).
     -- __iter__() instead of returning a permutation of range(n), it gives n random numbers each in range(n).
    """

    def __init__(self, length):
        self.length = length

    def __iter__(self):
        return iter(np.random.choice(self.length, size = self.length))

    def __len__(self):
        return self.length

def load_batch(loader, cuda = False, only_one_epoch = False):
    """This function loads a single batch with torch.utils.data.DataLoader and RandomSampler.

    By default, there is no end to the number of batches (no concept of epoch).
    This is overridden with only_one_epoch = True.
    """

    while True:
        for data, target in iter(loader):
            if  cuda: data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            yield data, target

        if only_one_epoch: break  # exit the loop if only_one_epoch == True


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

def do_reheating_cycle(lrs, bss, network_parameters, trainset, preparation_times, relaxation_time, OUTPUT_DIR):
    """This function performs a cold minimization, then it reheats the system at
    some given temperatures and saves losses and states in the given files.

     -- lrs, bss,           these are the sets of LRs and BSs that specify the temperature
     -- preparation_times,  times at which reheating starts (SORTED)
     -- relaxation_time,    training time after reheating
     -- OUTPUT_DIR          directory to save the output data (it is created in case)
    """

    # create folder fot output data, if it does not exist
    if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    cold_model = SimpleNet(*network_parameters)
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
            reheated_model = SimpleNet(*network_parameters)
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


# input_channels, output_classes, image_size (Fashion-MNIST = 28x28 -> size = 28)
network_parameters = (1, 10, 28)

# Temperatures for reheating; first one is for the cold model:
temps = [0.0002, 0.0006, 0.001]

# Duration of training: at each time, a bunch of reheated copies of the system
# are trained for a time relaxation_time.
# NOTE (important): times are expressed as BATCH TIME * LR!
ref_lr = 0.01
preparation_times = [ ref_lr*int(t) for t in [1e5, 5e5, 1e6] ]
relaxation_time = ref_lr*1e6


# --  Fixed BS  -------------------------------------------------------------- #


#print("Training at fixed BS:")
#bss = [32]*len(temps)  # lr = temp*bs, for temp in temps
#lrs = [bs*temp for temp, bs in zip(temps, bss)]
#do_reheating_cycle(
#    lrs, bss, network_parameters, trainset,
#    preparation_times = preparation_times,
#    relaxation_time = relaxation_time,
#    OUTPUT_DIR = 'reheating_same_sample_data/fixed_bs_cold_lr={}_bs={}'.format(lrs[0], bss[0])
#)


# --  Fixed LR  -------------------------------------------------------------- #


#print("Training at fixed LR:")
#lrs = [0.03]*len(temps)
#bss = [int(lr/temp) for temp, lr in zip(temps, lrs)]
#do_reheating_cycle(
#    lrs, bss, network_parameters, trainset,
#    preparation_times = preparation_times,
#    relaxation_time = relaxation_time,
#    OUTPUT_DIR = 'reheating_same_sample_data/fixed_lr_cold_lr={}_bs={}'.format(lrs[0], bss[0])
#)


# --  ANY  ------------------------------------------------------------------- #


lrs = [0.03, 0.03, 0.05]
bss = [150, 50, 50]
print("Training with temps = [" + ", ".join([ "{}/{}".format(lr, bs) for lr, bs in zip(lrs, bss) ]) + "]")
do_reheating_cycle(
    lrs, bss, network_parameters, trainset,
    preparation_times = preparation_times,
    relaxation_time = relaxation_time,
    OUTPUT_DIR = 'reheating_same_sample_data/fixed_lr_cold_lr={}_bs={}'.format(lrs[0], bss[0])
)

lrs = [0.03, 0.05]
bss = [50, 50]
print("Training with temps = [" + ", ".join([ "{}/{}".format(lr, bs) for lr, bs in zip(lrs, bss) ]) + "]")
do_reheating_cycle(
    lrs, bss, network_parameters, trainset,
    preparation_times = preparation_times,
    relaxation_time = relaxation_time,
    OUTPUT_DIR = 'reheating_same_sample_data/fixed_lr_cold_lr={}_bs={}'.format(lrs[0], bss[0])
)
