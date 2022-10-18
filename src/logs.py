import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


def show_progress(args, t, train_ts, train_loss, val_ts=None, val_loss=None, val_acc=None):
    clear_output(wait=True)
    with plt.ioff():
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(20, 5))
        fig.suptitle(f'Epoch {t:3.3f}', fontsize=16)
        ax1.set_title('loss')
        ax1.set_xlabel('time (epochs)')
        ax1.set_ylabel('loss')
        ax1.plot(train_ts, train_loss, c='darkblue', lw=3, label="Train loss")
        if val_ts is not None and len(val_ts):
            ax1.plot(val_ts, val_loss, c='green', marker='o', lw=3, label='Val loss')
            ax2.set_title('accuracy')
            ax2.set_xlabel('time (epochs)')
            ax2.plot(val_ts, val_acc, c='green', marker='o', lw=3)

        # plt.legend()
        plt.savefig(f'plots/g{t:.3f}_lr{args.lr}_mel{args.n_mels}_trunc{args.trunc}_epochs{args.epochs}SR_deepMFCC{args.n_mfcc}.png')
