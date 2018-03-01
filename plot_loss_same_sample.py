import os
import fnmatch
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib.backends.backend_pdf import PdfPages
import decimal

def get_lrbs_from_file(filename):
    """Extract LR and BS from a file's name.

    The filename has the format ''.../..._lr={LR}_bs={BS}.{extension}.'
    """

    lr, bs = tuple(os.path.splitext(filename)[0].split('_')[-2:])
    lr = decimal.Decimal(lr.split('=')[-1])
    bs = decimal.Decimal(bs.split('=')[-1])
    return lr, bs

def get_tlrbs_from_file(filename):
    """Extract TIME, LR and BS from a file's name.

    The filename has the format ''.../..._time={TIME}_lr={LR}_bs={BS}.{extension}.'
    """

    time, lr, bs = tuple(os.path.splitext(filename)[0].split('_')[-3:])
    time = decimal.Decimal(time.split('=')[-1])
    lr = decimal.Decimal(lr.split('=')[-1])
    bs = decimal.Decimal(bs.split('=')[-1])
    return time, lr, bs

def pickle_everything(filename):
    """This function opens the file and loads all the pickles inside, until it
    is empty (EOF).

    Pickled objects are returned in a list.
    """

    losses = []

    with open(filename, 'rb') as dump:
        losses = []

        while True:
            try:
                losses.append(pickle.load(dump))
            except EOFError:
                break

    return losses


# --  Load the loss data  ---------------------------------------------------- #


DATA_DIR = 'reheating_same_sample_data'

for directory in os.listdir(DATA_DIR):
    if not fnmatch.fnmatch(directory, '*_cold_lr=*_bs=*'): continue
    OUTPUT_DIR = DATA_DIR + '/' + directory

#    COLD_LR, COLD_BS, SIMULATION_TYPE = '0.0256', '128', 'fixed_bs'
#    COLD_LR, COLD_BS, SIMULATION_TYPE = '0.03', '150', 'fixed_lr'
#    OUTPUT_DIR = 'reheating_data/' + SIMULATION_TYPE + '_cold_lr=' + COLD_LR + '_bs=' + COLD_BS

    cold_losses = []
    all_reheated_losses = []

    print("Loading data in '{}'...".format(OUTPUT_DIR))
    for filename in os.listdir(OUTPUT_DIR):
        if fnmatch.fnmatch(filename, 'cold_losses_*.p'):
            lr, bs = get_lrbs_from_file(filename)
            losses = pickle_everything(OUTPUT_DIR + '/' + filename)
            cold_losses = [lr, bs, losses]

        elif fnmatch.fnmatch(filename, 'reheated_losses_*.p'):
            time, lr, bs = get_tlrbs_from_file(filename)
            losses = pickle_everything(OUTPUT_DIR + '/' + filename)
            all_reheated_losses.append([time, lr, bs, losses])

    if len(cold_losses) == 0:
        print("Cold losses not found!")
        quit()

    # Sort wrt to temperatures
    all_reheated_losses = sorted(all_reheated_losses, key = lambda el: el[1]/el[2])

################################################################################
#    print([ t for t,_ in cold_losses[-1] ])
#    for r in all_reheated_losses:
#        print(r[1]/r[2], r[0], [ t for t,_ in r[-1] ])
################################################################################

    # --  Plot the loss data  ------------------------------------------------ #


    # Plot all curves on top of each other in the same plot.
    # Color code associated with the temperature.
    temps_to_colors = { str(temp): idx + 1 for idx, temp in enumerate({ lr/bs for time, lr, bs, losses in all_reheated_losses }) }
    plt.figure(figsize = (16.8, 10))
    plt.tight_layout()

    print("Plotting cold run...")
    lr, bs = cold_losses[0:2]
    # When times were saved they have already been multiplied by the LR
    plt.plot([ t for t, l in cold_losses[-1] ], [ l for t, l in cold_losses[-1] ], '-',
        label = "T={:.2} (lr={}, bs={}),  time={}".format(lr/bs, lr, bs, 0),
        color = 'C0'
    )

    print("Plotting reheated losses...")
    for reheated_losses in all_reheated_losses:
        if len(reheated_losses[-1]) == 0: continue  # the simulation is still running

        time, lr, bs = reheated_losses[0:3]
        if not reheated_losses[-1][-1][1] > 0:
            print("Skipping lr={}, bs={} -- diverged".format(lr, bs))
            continue

        # When times were saved they have already been multiplied by the LR
        plt.plot(
            [ t for t, l in reheated_losses[-1] ], [ l for t, l in reheated_losses[-1] ], '-',
            label = "T={:.2} (lr={}, bs={}),  time={}".format(lr/bs, lr, bs, time),
            color = 'C' + str(temps_to_colors[str(lr/bs)])
        )

    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Loss')
    plt.xlabel('time ($\lambda\\times$Batch step)')
    plt.legend()

    #plt.show()
    with PdfPages(OUTPUT_DIR + '/all_losses.pdf') as pdf:
        pdf.savefig()

    plt.close()
