import os
import numpy as np
import antropy as ant
import sys
import warnings
import mne

from scipy import integrate, stats
from tqdm import tqdm


class StatisticalDecomposition:
    def __init__(self):

        self.delta = False
        self.delta_delta = False
        self.w_len = 1

    # Features used in classification:

    def f_mean(self, x):
        return np.mean(x)

    def f_absmean(self, x):
        return np.mean(np.abs(x))

    def f_maximum(self, x):
        return np.max(x)

    def f_absmax(self, x):
        return np.max(np.abs(x))

    def f_minimum(self, x):
        return np.min(x)

    def f_absmin(self, x):
        return np.min(np.abs(x))

    def f_minplusmax(self, x):
        return np.max(x) + np.min(x)

    def f_maxminusmin(self, x):
        return np.max(x) - np.min(x)

    def f_curvelength(self, x):

        cl = 0
        for i in range(x.shape[0] - 1):
            cl += abs(x[i] - x[i + 1])
        return cl

    def f_energy(self, x):
        return np.sum(np.multiply(x, x))

    def f_nonlinear_energy(self, x):
        # NLE(x[n]) = x**2[n] - x[n+1]*x[n-1]
        x_squared = x[1:-1] ** 2
        subtrahend = x[2:] * x[:-2]
        return np.sum(x_squared - subtrahend)

    # def f_ehf(x,prev):
    # (based on Basar et. al. 1983)
    # "prev" is array of values from prior context
    # rms = np.sqrt(np.mean(prev**2))
    # return 2*np.sqrt(2)*(max(x)/rms)

    def f_spec_entropy(self, x):
        return ant.spectral_entropy(x, self.fs, method="welch", normalize=True)

    def f_integral(self, x):
        return integrate.simps(x)

    def f_stddeviation(self, x):
        return np.std(x)

    def f_variance(self, x):
        return np.var(x)

    def f_skew(self, x):
        return stats.skew(x)

    def f_kurtosis(self, x):
        return stats.kurtosis(x)

    def f_sum(self, x):
        return np.sum(x)

    # Added ones: some of these are drawn from the antropy library: https://github.com/raphaelvallat/antropy

    def f_sample_entropy(self, x):
        x = x.astype(np.float64)
        return ant.sample_entropy(x, order=2, metric='chebyshev')

    def f_perm_entropy(self, x):
        return ant.perm_entropy(x, order=3, normalize=True)

    def f_svd_entropy(self, x):
        return ant.svd_entropy(x, order=3, delay=1, normalize=True)

    def f_app_entropy(self, x):
        return ant.app_entropy(x, order=2, metric='chebyshev')

    def f_petrosian(self, x):
        return ant.petrosian_fd(x)

    def f_katz(self, x):
        return ant.katz_fd(x)

    def f_higuchi(self, x):
        x = x.astype(np.float64)
        return ant.higuchi_fd(x, kmax=10)

    def f_rootmeansquare(self, x):
        return np.sqrt(np.mean(x ** 2))

    def f_dfa(self, x):
        x = x.astype(np.float64)
        return ant.detrended_fluctuation(x)

    def window_data(self, data: np.ndarray):

        # window the data using a stride length of half the window
        # (AKA half the unit stride):

        w_len = int(self.fs / self.w_len)  # windows of half the sampling frequency (this could be an arg)
        stride = int(w_len // 2)  # stride of half the window (this could be an arg)

        return np.lib.stride_tricks.sliding_window_view(data, w_len, axis=1)[:, ::stride]

    def compute_feats(self, windowed_data: np.ndarray):

        # Takes data of shape (channels, windows, time (window_len)),
        # returns an array of shape (channels, windows, num. features)

        funclist = [  # doesn't contain EHF since that must be added later
            self.f_mean,
            self.f_absmean,
            self.f_maximum,
            self.f_absmax,
            self.f_minimum,
            self.f_absmin,
            self.f_minplusmax,
            self.f_maxminusmin,
            self.f_curvelength,
            self.f_energy,
            self.f_nonlinear_energy,
            self.f_integral,
            self.f_stddeviation,
            self.f_variance,
            self.f_skew,
            self.f_kurtosis,
            self.f_sum,
            self.f_spec_entropy,
            self.f_sample_entropy,
            self.f_perm_entropy,
            self.f_svd_entropy,
            self.f_app_entropy,
            self.f_petrosian,
            self.f_katz,
            self.f_higuchi,
            self.f_rootmeansquare,
            self.f_dfa,
        ]

        channels, windows, time = np.shape(windowed_data)
        decomposed_data = np.empty((channels, windows, len(funclist)))

        for i in range(len(funclist)):
            decomposed_data[:, :, i] = np.apply_along_axis(funclist[i], 2, windowed_data)

        return decomposed_data

    def add_deltas(self, feats_array: np.ndarray):

        if self.delta_delta:
            deltas = np.diff(feats_array, axis=1)
            double_deltas = np.diff(deltas, axis=1)
            feats_array = np.hstack((feats_array[:, 2:, :], deltas[:, 1:, :], double_deltas))

        elif self.delta:
            deltas = np.diff(feats_array, axis=1)
            feats_array = np.hstack((feats_array[:, 1:, :], deltas))

        else:
            pass

        # resize (channels, windows, num. features) as (channels, windows*num. features):
        return feats_array.reshape(feats_array.shape[0], -1)

    def process_epochs(self, epoch):

        epoch = self.window_data(epoch)
        epoch = self.compute_feats(epoch)
        epoch = self.add_deltas(epoch)

        return epoch

    def get_feats_list(self):

        feats_list = [i[2:] for i in dir(self) if i[:2] == 'f_']
        num_feats = len(feats_list)
        deltas_diff = 0
        if self.delta:
            # lose 1 val either side for deltas:
            feats_list += ["d_" + i[2:] for i in dir(self) if i[:2] == 'f_']
            deltas_diff += 2
        if self.delta_delta:
            # lose 2 vals either side for delta-deltas
            feats_list += ["dd_" + i[2:] for i in dir(self) if i[:2] == 'f_']
            deltas_diff += 4

        return num_feats, feats_list, deltas_diff

    def main(self, epochs_fname):

        # Don't attempt delta_deltas if deltas are not active:
        if self.delta_delta:
            if not self.delta:
                self.delta_delta = False

        mne.utils.set_config('MNE_USE_CUDA', 'true')
        mne.cuda.init_cuda(verbose=True)
        epochs = mne.read_epochs(epochs_fname, verbose='WARNING')

        # ignore a warning generated by sample_entropy:
        if not sys.warnoptions:
            warnings.simplefilter("ignore")

        self.fs = epochs.info['sfreq']
        w_len = int(self.fs / self.w_len)  # windows of half the sampling frequency (this could be an arg)
        stride = int(w_len // 2)  # stride of half the window (this could be an arg)

        # work out how many window of data we'd have, given the shape of the data:
        print("shape ", epochs.get_data().shape)
        exemplars, channels, time = epochs.get_data().shape
        windows = len(np.lib.stride_tricks.sliding_window_view(range(time), w_len)[::stride])
        num_feats, feats_list, deltas_diff = self.get_feats_list()
        feats_diff = num_feats * deltas_diff
        num_feats = (windows * len(feats_list)) - feats_diff  # final automated calculation

        decomposed_data = np.empty((exemplars, channels, num_feats))

        for i in tqdm(range(exemplars)):
            decomposed_data[i] = self.process_epochs(epochs.next())

        decomposed_data = decomposed_data.reshape(decomposed_data.shape[0], -1)
        warnings.simplefilter("default")

        return decomposed_data

def extract_data_from_subject(root_dir, N_S, datatype):
    """
    Load all blocks for one subject and stack the results in X
    @author: Nicolás Nieto - nnieto@sinc.unl.edu.ar
    """
    import mne
    import numpy as np

    data = dict()
    y = dict()
    N_B_arr = [1, 2, 3]
    for N_B in N_B_arr:

        # name correction if N_Subj is less than 10
        if N_S < 10:
            Num_s = 'sub-0' + str(N_S)
        else:
            Num_s = 'sub-' + str(N_S)

        file_name = root_dir + '/derivatives/' + Num_s + '/ses-0' + str(N_B) + '/' + Num_s + '_ses-0' + str(
            N_B) + '_events.dat'
        y[N_B] = np.load(file_name, allow_pickle=True)

        if datatype == "EEG" or datatype == "eeg":
            #  load data and events
            file_name = root_dir + '/derivatives/' + Num_s + '/ses-0' + str(N_B) + '/' + Num_s + '_ses-0' + str(
                N_B) + '_eeg-epo.fif'
            X = mne.read_epochs(file_name, verbose='WARNING')
            data[N_B] = X._data

        elif datatype == "EXG" or datatype == "exg":
            file_name = root_dir + '/derivatives/' + Num_s + '/ses-0' + str(N_B) + '/' + Num_s + '_ses-0' + str(
                N_B) + '_exg-epo.fif'
            X = mne.read_epochs(file_name, verbose='WARNING')
            data[N_B] = X._data

        elif datatype == "Baseline" or datatype == "baseline":
            file_name = root_dir + '/derivatives/' + Num_s + '/ses-0' + str(N_B) + '/' + Num_s + '_ses-0' + str(
                N_B) + '_baseline-epo.fif'
            X = mne.read_epochs(file_name, verbose='WARNING')
            data[N_B] = X._data

        else:
            print("Invalid Datatype")

    X = np.vstack((data.get(1), data.get(2), data.get(3)))

    Y = np.vstack((y.get(1), y.get(2), y.get(3)))

    return X, Y


def select_time_window(X, t_start=1, t_end=2.5, fs=256):
    import numpy as np

    t_max = X.shape[2]
    start = max(round(t_start * fs), 0)
    end = min(round(t_end * fs), t_max)

    # Copy interval
    X = X[:, :, start:end]
    return X


def filter_by_condition(X, Y, condition):
    if not condition:
        raise Exception("You have to select the conditions!")

    if condition.upper() == "ALL":
        return X, Y
    else:
        X_r = []
        Y_r = []
        if condition.upper() == "PRON" or condition.upper() == "PRONOUNCED":
            p = 0
        elif condition.upper() == "IN" or condition.upper() == "INNER":
            p = 1
        elif condition.upper() == "VIS" or condition.upper() == "VISUALIZED":
            p = 2
        else:
          raise Exception("The condition " + condition + " doesn't exist!")

        X_r = X[Y[:,2] == p]
        Y_r = Y[Y[:,2] == p]

    return X_r, Y_r


def get_subjects_data_and_label(root_dir, condition, t_start=1, t_end=2.5, fs=256):
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    datatype = "EEG"
    data = []
    labels = []

    for subject in subjects:
        X, Y = extract_data_from_subject(root_dir, subject, datatype)
        X = select_time_window(X=X, t_start=t_start, t_end=t_end, fs=fs)
        X, Y = filter_by_condition(X, Y, condition)
        data.append(X)
        labels.append(Y.T[1])

    return data, labels


def get_subjects_data_label_group(root_dir, condition, t_start=1, t_end=2.5, fs=256):
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    datatype = "EEG"
    data = []
    labels = []
    groups = []

    for subject in subjects:
        X, Y = extract_data_from_subject(root_dir, subject, datatype)
        X = select_time_window(X=X, t_start=t_start, t_end=t_end, fs=fs)
        X, Y = filter_by_condition(X, Y, condition)
        data.append(X)
        labels.append(Y.T[1])
        groups.append(np.full(len(X), subject))

    return data, labels, groups


def get_subjects_data_and_label2(root_dir, condition, t_start=1, t_end=2.5, fs=256):
    subjects = [1]
    data = []
    labels = []

    for subject in subjects:
        X, Y = extract_data_from_subject2(root_dir, subject)
        X, Y = filter_by_condition(X, Y, condition)
        data.append(X)
        labels.append(Y.T[1])

    return data, labels


def get_one_subject_data_and_label(root_dir, subject, condition, t_start=1, t_end=2.5, fs=256):
    datatype = "EEG"
    data = []
    labels = []

    X, Y = extract_data_from_subject(root_dir, subject, datatype)
    X = select_time_window(X=X, t_start=t_start, t_end=t_end, fs=fs)
    X, Y = filter_by_condition(X, Y, condition)
    data.append(X)
    labels.append(Y.T[1])

    return data, labels


def extract_data_from_subject2(root_dir, N_S):
    """
    Load all blocks for one subject and stack the results in X
    @author: Nicolás Nieto - nnieto@sinc.unl.edu.ar
    """
    import mne
    import numpy as np

    data = dict()
    y = dict()
    N_B_arr = [1, 2, 3]
    for N_B in N_B_arr:

        # name correction if N_Subj is less than 10
        if N_S < 10:
            Num_s = 'sub-0' + str(N_S)
        else:
            Num_s = 'sub-' + str(N_S)

        file_name = root_dir + '/derivatives/' + Num_s + '/ses-0' + str(N_B) + '/' + Num_s + '_ses-0' + str(
            N_B) + '_events.dat'
        y[N_B] = np.load(file_name, allow_pickle=True)

        file_name = root_dir + '/derivatives/' + Num_s + '/ses-0' + str(N_B) + '/' + Num_s + '_ses-0' + str(
            N_B) + '_eeg-epo.fif'
        data[N_B] = StatisticalDecomposition().main(file_name)

    X = np.vstack((data.get(1), data.get(2), data.get(3)))
    Y = np.vstack((y.get(1), y.get(2), y.get(3)))

    return X, Y
