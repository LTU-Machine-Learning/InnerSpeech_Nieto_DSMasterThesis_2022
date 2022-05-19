import numpy as np
from scipy import integrate
from scipy import stats
import antropy as ant


# Define all the features
def f_mean(x):
    return np.mean(x, axis=-1)


def f_std(x):
    return np.std(x, axis=-1)


def f_ptp(x):
    return np.ptp(x, axis=-1)


def f_var(x):
    return np.var(x, axis=-1)


def f_minim(x):
    return np.min(x, axis=-1)


def f_maxim(x):
    return np.max(x, axis=-1)


def f_argminim(x):
    return np. argmin(x, axis=-1)


def f_argmaxim(x):
    return np.argmax(x,axis=-1)


def f_rms(x):
    return np.sqrt(np.mean(x**2, axis=-1))


def f_abs_diff_signal(x):
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)


def f_skewness(x):
    return stats.skew(x, axis=-1)


def f_kurtosis(x):
    return stats.kurtosis(x, axis=-1)


def f_minplusmax(x):
    return np.max(x, axis=-1) + np.min(x, axis=-1)


def f_maxminusmin(x):
    return np.max(x, axis=-1) - np.min(x, axis=-1)


def f_integral(x):
    return integrate.simps(x, axis=-1)


def f_petrosian(x):
    return ant.petrosian_fd(x, axis=-1)


def f_katz(x):
    return ant.katz_fd(x, axis=-1)


def generate_features(data_array, features_list):
    features = []

    for d in data_array:
        calculated = []
        for feature in features_list:
            calculated.append(feature(d))
        features.append(np.concatenate(calculated, axis=None))

    return np.array(features)