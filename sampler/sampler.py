import numpy as np

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
