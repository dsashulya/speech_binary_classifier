import numpy as np
from tqdm import tqdm
import librosa
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset


class MelSpecs(Dataset):
    """
    used for CNN training
    """
    def __init__(self, data: Dataset, labels: np.ndarray, sr: int = 24000,
                 trunc=0, n_fft=2048, hop_length=512, n_mels=128):
        """
        trunc: either 0 for no truncation or the rightmost index to be left of the signal
        """
        self.data = data
        self.labels = labels
        self.sr = sr
        self.trunc = trunc
        self.melspecobj = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=n_mels,
            mel_scale="htk",
        )

        self.n_mels = n_mels
        self.n_wins = int(trunc / hop_length) - 1

    def __getitem__(self, n: int):
        signal = self.data[n][0].squeeze(axis=0)

        if not self.trunc:
            return self.melspecobj(signal)[None, :, :], self.labels[n]

        # padding shorter signals
        out = torch.zeros(self.trunc)
        out[:len(signal[:self.trunc])] = signal[:self.trunc]
        return self.melspecobj(out)[None, :, :], self.labels[n]

    def get_label(self, n: int):
        return self.labels[n]

    def size(self, i: int):
        s = (len(self.data), self.n_mels, self.n_wins)
        assert i < len(s), "Index out of range"
        return s[i]

    def __len__(self):
        return len(self.data)


class MFCC(MelSpecs):
    def __init__(self, data: Dataset, labels: np.ndarray, sr: int = 24000,
                 trunc=0, n_fft=2048, hop_length=512, n_mfcc=40, n_mels=128):
        super().__init__(data, labels,
                         sr=sr, trunc=trunc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.data = data
        self.labels = labels
        self.sr = sr
        self.trunc = trunc
        self.mfccobj = T.MFCC(sample_rate=sr, n_mfcc=n_mfcc,
                              melkwargs={"n_mels": n_mels,
                                         "n_fft": n_fft,
                                         "hop_length": hop_length})

        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_wins = int(trunc / hop_length) - 1

    def __getitem__(self, n: int) -> np.ndarray:
        signal = self.data[n][0]

        if not self.trunc:
            return self.mfccobj(signal), self.labels[n]

        # padding shorter signals
        signal = signal.squeeze(axis=0)
        out = torch.zeros(self.trunc)
        out[:len(signal[:self.trunc])] = signal[:self.trunc]
        return self.mfccobj(out)[None, :, :], self.labels[n]

    def size(self, i: int):
        s = (len(self.data), self.n_mfcc, self.n_wins)
        assert i < len(s), "Index out of range"
        return s[i]


class LibriDataset(Dataset):
    """
    used for SVM training
    """
    def __init__(self, data: Dataset, labels: np.ndarray, sr: int = 24000,
                 trunc=0):
        self.data = data
        self.labels = labels
        self.sr = sr
        self.trunc = trunc

    def __getitem__(self, n: int):
        signal = self.data[n][0].squeeze(axis=0)

        if not self.trunc:
            return signal.numpy(), self.labels[n]

        # padding shorter signals
        out = torch.zeros(self.trunc)
        out[:len(signal[:self.trunc])] = signal[:self.trunc]
        return out.numpy(), self.labels[n]

    def get_label(self, n: int):
        return self.labels[n]

    def __len__(self):
        return len(self.data)


def create_data_matrix(data, trunc, sr=24000, mode='dft', max_len=12000, n_mels=128, n_mfcc=128,
                       n_fft=2048, hop_length=512):
    n_wins = int(trunc / hop_length) + 1
    if mode == 'mels':
        max_len = n_mels * n_wins
    elif mode == 'mfcc':
        max_len = n_mfcc * n_wins
    X = np.zeros((len(data), max_len), dtype=float)
    y = np.zeros(len(data), dtype=int)

    win = librosa.core.spectrum.get_window('hann', trunc, fftbins=True)
    for i, (sig, label) in enumerate(tqdm(data)):
        y[i] = label
        if mode == 'dft':
            X[i, :] = abs(np.fft.fft(sig * win))[:max_len]
        elif mode == 'mels':
            X[i, :] = librosa.power_to_db(
                librosa.feature.melspectrogram(y=sig[:trunc],
                                               sr=sr,
                                               n_mels=n_mels,
                                               fmin=1,
                                               fmax=8192,
                                               n_fft=n_fft, hop_length=hop_length,
                                               pad_mode='constant')).flatten()
        elif mode == 'mfcc':
            X[i, :] = librosa.feature.mfcc(
                y=sig[:trunc], sr=sr, n_mfcc=n_mfcc, fmin=1, fmax=8192, n_fft=n_fft, hop_length=hop_length,
                pad_mode='constant').flatten()
    return X, y
