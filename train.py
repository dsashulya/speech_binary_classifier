import argparse
import pickle

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from src.data import MelSpecs, MFCC, LibriDataset, create_data_matrix
from src.logs import show_progress
from src.models import CNN

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='SVM', type=str, required=True,
                        help="Specify one model from [SVM, CNN]")

    # DATA
    parser.add_argument('--seed', default=10, type=int, required=False,
                        help="Seed used for train/val/test split")
    parser.add_argument('--download', default=False, type=lambda x: bool(int(x)), required=False,
                        help='Whether to download LibriTTS')
    parser.add_argument('--path_to_data', default='data/', type=str, required=False,
                        help="Path to LibriTTS in a local folder")
    parser.add_argument('--url', default='dev-clean', type=str, required=False,
                        help="Dataset version to download (dev-clean by default)")
    parser.add_argument('--path_to_labels', default='data/genders', type=str, required=True,
                        help="Path to data labels in pickle format")
    parser.add_argument('--trunc', default=192000, type=int, required=False,
                        help='Maximum singnal length')
    parser.add_argument('--n_mels', default=128, type=int, required=False,
                        help='Number of Mels used for Mel Spectrogram')
    parser.add_argument('--n_fft', default=2048, type=int, required=False,
                        help="Num fft for Mel Specs")
    parser.add_argument('--hop_length', default=512, type=int, required=False,
                        help="Hop length for Mel Specs")
    parser.add_argument('--sr', default=24000, type=int, required=False,
                        help="Data sample rate")
    parser.add_argument('--svm_mode', default='mfcc', type=str, required=False,
                        help="Data augmentation mode for SVM")
    parser.add_argument('--dft_max_len', default=12000, type=int, required=False,
                        help="Used for SVM in DFT mode")
    parser.add_argument('--mfcc', default=False, type=lambda x: bool(int(x)), required=False,
                        help="Use MFCC instead of Mel Specs")
    parser.add_argument('--n_mfcc', default=40, type=int, required=False,
                        help='Number of MFCC used for MFCC Spectrogram')

    # TRAINING
    parser.add_argument('--device', default='cuda', type=str, required=False,
                        help="Device to train CNN on in [cuda, cpu]")
    parser.add_argument('--train_percent', default=0.6, type=float, required=False,
                        help="% of data used for training")
    parser.add_argument('--val_percent', default=0.2, type=float, required=False,
                        help="% of data used for evaluating")
    parser.add_argument('--batch_size', default=32, type=int, required=False,
                        help="Batch size used for training CNN")
    parser.add_argument('--shuffle', default=True, type=lambda x: bool(int(x)), required=False,
                        help="DataLoader shuffle parameter")
    parser.add_argument('--epochs', default=5, type=int, required=False,
                        help="Num epochs for training CNN")
    parser.add_argument('--lr', default=1e-3, type=float, required=False,
                        help="CNN Adam learning rate")
    parser.add_argument('--weight_decay', default=5e-4, type=float, required=False,
                        help="CNN Adam weight decay")
    parser.add_argument('--log_every', default=20, type=int, required=False,
                        help="Num training steps before logging loss and accuracy")
    parser.add_argument('--do_eval', default=True, type=lambda x: bool(int(x)), required=False,
                        help="Whether to evaluate model on Val Dataset")
    parser.add_argument('--graph', default=True, type=lambda x: bool(int(x)), required=False,
                        help="Whether to make loss and accuracy graphs at the end of the training")

    return parser


def load_data(args):
    print("Loading data...")
    data = torchaudio.datasets.LIBRITTS(args.path_to_data, url=args.url, download=args.download)

    with open(args.path_to_labels, "rb") as file:
        genders = pickle.load(file)
    return data, genders


def train_test_split(args, dataset):
    train_num = round(len(dataset) * args.train_percent)
    val_num = round(len(dataset) * args.val_percent)
    test_num = len(dataset) - train_num - val_num

    if args.model.lower() == 'svm':
        train_num += val_num
        val_num = 0

    return random_split(dataset, [train_num, val_num, test_num],
                        generator=torch.Generator().manual_seed(args.seed))


def train_CNN(args):
    if torch.cuda.is_available() and args.device.lower() == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    args.device = device

    data, genders = load_data(args)

    if args.mfcc:
        dataset = MFCC(data, genders,
                       sr=args.sr,
                       n_fft=args.n_fft,
                       hop_length=args.hop_length,
                       trunc=args.trunc,
                       n_mels=args.n_mels,
                       n_mfcc=args.n_mfcc)
    else:
        dataset = MelSpecs(data, genders,
                           sr=args.sr,
                           n_fft=args.n_fft,
                           hop_length=args.hop_length,
                           trunc=args.trunc,
                           n_mels=args.n_mels)

    train, val, test = train_test_split(args, dataset)

    train_dl = DataLoader(train, shuffle=args.shuffle, batch_size=args.batch_size)
    val_dl = DataLoader(val, shuffle=args.shuffle, batch_size=args.batch_size)
    test_dl = DataLoader(test, shuffle=args.shuffle, batch_size=args.batch_size)

    model = CNN().to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = nn.CrossEntropyLoss()

    print("Starting training...")
    model.train()
    running_loss = 0.
    losses, train_ts, val_losses, val_accs = [], [], [], []
    num_batches = len(train_dl)
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        epoch_loss, epoch_acc = 0., 0.
        total = 0
        for i, batch in enumerate(tqdm(train_dl)):
            x, target = batch
            out = model(x.to(args.device)).cpu()

            optimizer.zero_grad()
            ll = loss(out, target)
            ll.backward()
            optimizer.step()

            epoch_loss += ll.item()
            running_loss += ll.item()

            out_labels = F.softmax(out, dim=1).argmax(axis=1)
            epoch_acc += torch.sum(out_labels == target)

            total += len(x)

            if (i + 1) % args.log_every == 0 or (i + 1) == num_batches and (epoch + 1) == args.epochs:
                t = epoch + (i + 1) / num_batches
                train_ts.append(t)
                losses.append(running_loss / args.log_every)
                running_loss = 0.

                if args.do_eval:
                    val_loss, val_acc = eval_CNN(args, model, val_dl, epoch=epoch, loss=loss, test=False)
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)

                show_progress(args, t, train_ts, losses, train_ts, val_losses, val_accs)

    print("Finished training, starting testing...")
    eval_CNN(args, model, test_dl, test=True)


def eval_CNN(args, model, data, epoch=None, loss=None, test=False):
    model.eval()
    val_loss, val_acc = 0., 0.
    total = 0
    for batch in tqdm(data):
        x, target = batch
        out = model(x.to(args.device)).cpu()
        if loss:
            ll = loss(out, target)
            val_loss += ll.item()

        out_labels = F.softmax(out, dim=1).argmax(axis=1)
        val_acc += torch.sum(out_labels == target)
        total += len(x)
    if not test:
        print(f"Epoch {epoch + 1} VAL Loss = {val_loss / len(data):.3f} VAL Accuracy = {val_acc / total:.3f}")
        return val_loss / len(data), val_acc / total
    else:
        print(f"TEST Accuracy = {val_acc / total:.3f}")
        return val_acc / total


def train_SVM(args):
    data, genders = load_data(args)
    ds = LibriDataset(data, genders, trunc=args.trunc)
    train, _, test = train_test_split(args, ds)
    X_train, y_train = create_data_matrix(train, args.trunc,
                                          sr=args.sr,
                                          mode=args.svm_mode.lower(),
                                          max_len=args.dft_max_len,
                                          n_mels=args.n_mels, n_mfcc=args.n_mfcc,
                                          n_fft=args.n_fft, hop_length=args.hop_length)
    X_test, y_test = create_data_matrix(test, args.trunc,
                                        sr=args.sr,
                                        mode=args.svm_mode.lower(),
                                        max_len=args.dft_max_len,
                                        n_mels=args.n_mels, n_mfcc=args.n_mfcc,
                                        n_fft=args.n_fft, hop_length=args.hop_length)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', verbose=True))
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print(f"SVM {args.svm_mode} TEST ACCURACY = {sum(pred == y_test) / len(pred):.3f}")


if __name__ == "__main__":
    args = setup_argparser().parse_args()
    assert args.model.lower() in ['cnn', 'svm'], "Model not supported"
    assert args.device.lower() in ['cuda', 'cpu'], "Device not recognised"

    if args.model == 'CNN':
        train_CNN(args)
    else:
        assert args.svm_mode.lower() in ['dft', 'mels', 'mfcc'], "Unkown data augmentation mode"
        train_SVM(args)
