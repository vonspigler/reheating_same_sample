import os
import fnmatch
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib.backends.backend_pdf import PdfPages
import decimal
from collections import OrderedDict

class MultidimDict(OrderedDict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]

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

    print(" -- Loading data in '{}'...".format(OUTPUT_DIR))
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

    # Sort wrt to temperatures, then wrt to times
    all_reheated_losses = sorted(all_reheated_losses,
        key = lambda el: (el[1]/el[2], el[0])
    )

    # Sets with all the temperatures and all the times (unique):
    all_temps = sorted(list({ str(lr/bs) for time, lr, bs, losses in all_reheated_losses }), key = float)
    all_times = sorted(list({ str(time) for time, lr, bs, losses in all_reheated_losses }), key = float)

    # map_reheated_losses['temp']['time'] = (lr, bs, losses)
    # best used in combination of all_temps/times
    map_reheated_losses = MultidimDict()
    for time, lr, bs, losses in all_reheated_losses:
        map_reheated_losses[str(lr/bs)][str(time)] = (lr, bs, losses)


    # --  Plot the loss data  ------------------------------------------------ #
    ####  TO BE MODIFIED NOW THAT I HAVE map_reheated_losses  ##################


    # Plot all curves on top of each other in the same plot.
    # Color code associated with the temperature.
    # Width and transparency are large for short reheated preparation times
    temps_to_colors = { str(temp): idx + 1 for idx, temp in enumerate(sorted({ lr/bs for time, lr, bs, losses in all_reheated_losses })) }
    times_to_index = { str(time): idx + 1 for idx, time in enumerate(sorted({ time for time, lr, bs, losses in all_reheated_losses })) }
    alphas = list(reversed([ a/len(times_to_index) for a in range(len(times_to_index), 0, -1) ]))
    times_to_alpha = { key: alphas[value - 1] for key, value in times_to_index.items() }
    widths = list([ a for a in range(len(times_to_index), 0, -1) ])
    times_to_width = { key: widths[value - 1] for key, value in times_to_index.items() }

    pdf = PdfPages(OUTPUT_DIR + '/all_losses.pdf')

    plt.figure(figsize = (16.8, 10))
    plt.tight_layout()

    lr, bs = cold_losses[0:2]
    print("Plotting cold run: temp={}".format(lr/bs))
    # When times were saved they have already been multiplied by the LR
    plt.plot([ t for t, l in cold_losses[-1] ], [ l for t, l in cold_losses[-1] ], '-',
        label = "T={:.2} (lr={}, bs={}),  time={}".format(lr/bs, lr, bs, 0),
        color = 'C0'
    )

    for time in all_times:
        print("  Preparation time=" + time + ":\n    Temperatures: ", end = '')

        for temp in all_temps:
            if len(map_reheated_losses[temp][time]) == 0: continue
            lr, bs, losses = map_reheated_losses[temp][time]

            if not losses[-1][1] > 0:
                print("[{}:{}], ".format(lr, bs), end = '')
                continue
            else:
                print(temp + ", ", end = '')

            plt.plot(
                [ t for t, l in losses ], [ l for t, l in losses ], '-',
                label = "T={:.2} (lr={}, bs={}),  time={}".format(float(temp), lr, bs, time),
                color = 'C' + str(temps_to_colors[temp]),
                linewidth = times_to_width[time],
                alpha = times_to_alpha[time]
            )

        print()

    plt.ylim(1e-5, 10)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Loss')
    plt.xlabel('time ($\lambda\\times$Batch step)')
    plt.legend()

    #plt.show()
    pdf.savefig()
    print()

    # --  Other plots, for each preparation time  ---------------------------- #


    print("Plotting only reheated losses with shifted times:")
    for time in all_times:
        print("  Preparation time=" + time + ":\n    Temperatures: ", end = '')
        plt.figure(figsize = (16.8, 10))
        plt.tight_layout()

        plt.axhline(y = cold_losses[-1][0][1], linestyle = '--', color = 'black', label = 'Loss(time = 0)')

        lr, bs, losses = cold_losses
        if len([ t for t, l in losses if t > float(time) ]) > 0:
            print("[COLD], ", end = '')
            plt.plot(
                [ t - float(time) for t, l in losses ], [ l for t, l in losses ], '-',
                label = "T={:.2} (lr={}, bs={}),  time={}".format(float(lr/bs), lr, bs, time),
                color = 'C0'
            )

        for temp in all_temps:
            if len(map_reheated_losses[temp][time]) == 0: continue
            lr, bs, losses = map_reheated_losses[temp][time]

            if not losses[-1][1] > 0:
                print("[{}:{}], ".format(lr, bs), end = '')
                continue
            else:
                print(temp + ", ", end = '')

            plt.plot(
                [ t - float(time) for t, l in losses ], [ l for t, l in losses ], '-',
                label = "T={:.2} (lr={}, bs={}),  time={}".format(float(temp), lr, bs, time),
                color = 'C' + str(temps_to_colors[temp])
            )

        print()

        plt.ylim((1e-6,1e2))
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Loss')
        plt.xlabel('time ($\lambda\\times$Batch step) - prep_time (={})'.format(time))
        plt.legend()

        #plt.show()
        pdf.savefig()


    # --  Close everything  -------------------------------------------------- #


    pdf.close()
    plt.close()
    print()
