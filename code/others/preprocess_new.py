import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import antropy as ant
import subprocess
import pdb
import json
import argparse
import ast
import sys
import warnings
import xgboost
import random

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from scipy import integrate, stats

import mne
from mne import Epochs, pick_types, find_events, pick_types, set_eeg_reference
from mne.io import concatenate_raws, read_raw_bdf
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, corrmap

from autoreject import AutoReject
from autoreject import get_rejection_threshold
from autoreject import Ransac

# ar = AutoReject()
# rsc = Ransac()

from tqdm import tqdm
from pprint import pprint
from collections import defaultdict

from pdb import set_trace as db


class StatisticalDecomposition:
    def __init__(self, _):

        self.args = _
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

        epochs = mne.read_epochs(epochs_fname, preload=True, verbose=self.args['verbose'])
        labels = np.array([i for i in epochs._metadata['event_name']])

        X_fname = f"{epochs_fname.split('-')[0]}-decomp.npy"
        y_fname = f"{epochs_fname.split('-')[0]}-labels.npy"

        if self.args['load']:
            if os.path.exists(X_fname):
                return (np.load(X_fname), np.load(y_fname))
            else:
                print(f"Cannot load data: {X_fname} not found. Proceeding to compute statistical decomposition...")

        # ignore a warning generated by sample_entropy:
        if not sys.warnoptions:
            warnings.simplefilter("ignore")

        self.fs = epochs.info['sfreq']
        w_len = int(self.fs / self.w_len)  # windows of half the sampling frequency (this could be an arg)
        stride = int(w_len // 2)  # stride of half the window (this could be an arg)

        # work out how many window of data we'd have, given the shape of the data:
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

        if self.args['save']:
            np.save(X_fname, decomposed_data)
            np.save(y_fname, labels)

        return (decomposed_data, labels)


class Preprocess(StatisticalDecomposition):

    def __init__(self, _, paths,
                 # model_name = None,
                 # channels_transform = False,
                 # frequency_bands_transform = False,
                 ):

        super().__init__(_)
        self.args = _
        # self.model_name = model_name
        # self.channels_transform = channels_transform # either all channels, or selected channels around Broca's Wernicke's area
        # self.frequency_bands_transform = frequency_bands_transform # either all frequency bands, or high gamma only
        # self.window_transforms = {
        # 'CNN':lambda X,y : (X, y), # D x H x W = sensors x freqbands x samples

        # 1Hz HP --> Interpolate Bad channels --> Re-reference --> Notch --> Autoreject Epochs --> Re-reference --> ICA fit

    def remove_baseline_drift(self, data):
        ### Low-frequency drifts ###
        # Sanity check that our data does NOT have DC drift applied online (during data collection)

        if self.args['plot']:
            raw.plot(
                duration=60,
                remove_dc=False)  # CONFIRMED - DC drift was not automatically applied

        # Remove the lf components, if needed. AT LEAST need to do a high-pass for DC drift.
        data.filter(
            l_freq=1.,  # 1Hz is an excellent (and standard) highpass filter
            h_freq=None,
            method='fir',
            fir_window='hamming',
            fir_design='firwin',  # 'firwin2'
            verbose=self.args['verbose'],
            # l_trans_bandwidth=0.5,
            # h_trans_bandwidth=0.5,
            # filter_length='10s',
            # phase='zero-double',
        )

        return data

    def interpolate_bad_channels(self, epochs):

        ransac = Ransac(
            n_resample=50,
            min_channels=0.25,
            min_corr=0.75,
            unbroken_time=0.4,
            n_jobs=-1,
            random_state=123,
            picks=None,
            verbose="progressbar")

        ransac.fit(epochs)

        # If there are any bad channels, we'll drop them here for the benefit of making the ICA more robust:
        epochs.info['bads'] = ransac.bad_chs_
        epochs.interpolate_bads()  # the alternative is dropping: epochs.drop_bad(reject=reject)

        return ransac.bad_chs_, epochs

    def re_reference(self, data, ref_channels):

        # Set mastoid references:
        data.set_eeg_reference(
            ref_channels=ref_channels,
            # copy=False,
            projection=False,
            ch_type='auto',
            forward=None,
            verbose=self.args['verbose']
        )

        return data

    def remove_line_noise(self, raw, notches):

        ### Power line noise ###

        # Sanity check that our data actually does have line noise artifacts:

        if self.args['plot']:
            raw.plot_psd(
                area_mode='range',
                tmax=np.inf,
                picks='eeg',
                average=False)  # CONFIRMED - notch filter was not automatically applied

        # notch filter the data at 50, 100 and 150 to remove the 50 Hz line noise and its harmonics:

        raw.notch_filter(
            notches,
            picks='eeg',
            filter_length='auto',
            phase='zero-double',  # 'zero'
            fir_design='firwin')

        return raw

    def reject_bad_epochs(self, epochs):

        ar = AutoReject(
            n_interpolate=[1, 2, 3, 4],
            consensus=None,
            thresh_func=None,
            cv=10, picks=None,
            thresh_method='bayesian_optimization',
            n_jobs=-1,
            random_state=123,
            verbose="progressbar"
        )

        ar.fit(epochs)
        epochs, reject_log = ar.transform(epochs, return_log=True)

        if self.args['plot']:
            epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))  # visualize dropped epochs
            reject_log.plot('horizontal')  # see epochs marked as bad

        return epochs

    def fit_ICA(self, epochs, interpolated_channels=0):

        # ICA is not deterministic (e.g., the components may get a sign flip on different runs,
        # or may not always be returned in the same order), so we should set a seed.

        # recommended to reduce the dimensionality (by 1 for average reference and 1 for each
        # interpolated channel for optimal ICA performance, so set this here:
        # (remember average reference would reduce this by 1 again...)
        n = int(len(epochs._channel_type_idx['eeg']) - len(interpolated_channels))

        ica = mne.preprocessing.ICA(
            n_components=n,  # set to no. of data channels
            noise_cov=None,
            random_state=123,  # seed
            method='fastica',
            fit_params=None,
            max_iter='auto',
            allow_ref_meg=False,
            verbose=self.args['verbose'],
        )

        # fit to filt_raw, apply to raw. Note we're fitting the ICA to fixed-length epochs:
        # that's absolutely fine; it just provides a more robust ICA solution when we 
        # later apply it to the raw data.
        ica.fit(epochs,  # i.e. fit on all epochs not auto-rejected
                picks='data',
                start=None,
                stop=None,
                decim=None,  # every Nth sample
                reject=None,  # peak-to-peak amplitudes
                flat=None,
                tstep=2.0,
                reject_by_annotation=True,
                verbose=self.args['verbose'])

        if self.args['plot']:
            ica.plot_sources(raw)  # note applied to original raw
            ica.plot_components(inst=raw)
            # if we have evoked; good to sanity-check:
            # eog_evoked = create_eog_epochs(filt_raw).average()
            # eog_evoked.apply_baseline(baseline=(None, -0.2))
            # ica.plot_sources(eog_evoked) if we have evoked; good to sanity-check

        # MANUALLY EXCLUDE BASED ON VISUAL INSPECTION:
        # ica.exclude = [1] # IC 1 is EOG artifacts, for example
        # ica.apply(raw)

        # AUTOMATICALLY EXCLUDE BASED ON EOG CHANNELS:
        ica.exclude = []
        # find which ICs match the EOG pattern:
        # NOTE! if we expect saccades and blinks then we should be looking for at least 2 EOG ICs.
        eog_indices, eog_scores = ica.find_bads_eog(
            epochs,
            ch_name=None,
            threshold=3.0,
            start=None,
            stop=None,
            l_freq=1,
            h_freq=10,
            reject_by_annotation=True,
            measure='zscore',
            verbose=self.args['verbose'])

        if self.args['plot']:
            ica.plot_scores(eog_scores)  # barplot of ICA component "EOG match" scores
            ica.plot_properties(raw, picks=eog_indices)  # plot diagnostics

        # # find which ICs match the ECG pattern:
        # ecg_indices, ecg_scores = ica.find_bads_ecg(
        #     epochs,
        #     ch_name=None,
        #     threshold='auto',
        #     start=None,
        #     stop=None,
        #     l_freq=8,
        #     h_freq=16, 
        #     method='correlation',
        #     reject_by_annotation=True,
        #     measure='zscore', 
        #     verbose=self.args['verbose'])

        # if self.args['plot']:
        #     ica.plot_scores(ecg_scores) # barplot of ICA component "EOG match" scores
        #     ica.plot_properties(raw, picks=ecg_indices) # plot diagnostics

        ica.exclude = eog_indices  # + ecg_indices

        return ica

    def make_epochs_from_sliding_window(self, raw):

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=3,
            preload=True,
            reject_by_annotation=True,
            proj=False,
            overlap=0.0,
            verbose=self.args['verbose'],
        )  # This is preserve all of the data

        return epochs

    def make_epochs_from_events(self, raw, events, event_id, tmin, tmax):

        metadata, events, event_id = mne.epochs.make_metadata(
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            sfreq=raw.info['sfreq'],
            row_events=None,
            keep_first=None,
            keep_last=None,
        )

        ### Epochs ###

        # When accessing data, epochs are linearly detrended (changable - see below),
        # baseline-corrected and decimated, then projectors are (optionally) applied.

        # N.b. If wanting baseline interval as only one sample, we must use
        # `baseline=(0, 0)` as opposed to `baseline=(None, 0)`

        # ALSO Removing EOG artifacts with ICA likely introduces DC offsets,
        # so it's imperative to do baseline correction AFTER ICA and filtering.

        # mne.event.shift_time_events --- will need this for experiments.

        reject = dict(
            eeg=40e-6,  # unit: V (EEG channels)
            # eog=250e-6 # unit: V (EOG channels) (we remove these, so ignore)
        )

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,  # lazy defaults -.2
            tmax=tmax,  # lazy defaults .5
            baseline=(0, 1 / raw.info['sfreq'] * 2),  # in seconds. Take a couple of samples to be safe.
            picks=self.args['picks'],
            preload=True,
            # reject = reject, # leave for now
            flat=None,
            proj=False,
            decim=1,
            reject_tmin=None,
            reject_tmax=None,
            detrend=1,  # 0 = constant (DC) detrend, 1 = linear detrend
            on_missing='raise',
            reject_by_annotation=True,
            metadata=metadata,
            event_repeated='error',  # 'drop' for coarticulation: keep 1st phone only
            verbose=self.args['verbose'])

        # LEFT HERE FOR FUTURE REFERENCE: To label epochs with blinks you can do:
        #     eog_events = mne.preprocessing.find_eog_events(raw)  
        #     n_blinks = len(eog_events)  
        #     onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25  
        #     duration = np.repeat(0.5, n_blinks)  
        #     description = ['bad blink'] * n_blinks  
        #     annotations = mne.Annotations(onset, duration, description)  
        #     raw.set_annotations(annotations)             

        # evoked = epochs.average() # Only if you need the evoked object.

        return epochs

    def subband_decomposition(self, fmin, fmax):

        fname = "{}_eeg.fif.gz".format(self.args['id'])
        # (re)load the data to save memory
        raw = mne.io.read_raw_fif(
            fname,
            allow_maxshield=False,
            preload=True,
            on_split_missing='raise',
            verbose=self.args['verbose'],
        )

        # bandpass filter
        raw.filter(
            l_freq=fmin,
            h_freq=fmax,
            picks=None,
            filter_length='auto',
            l_trans_bandwidth=0.5,  # make sure filter params are the same
            h_trans_bandwidth=0.5,
            n_jobs=-1,  # use more jobs to speed up
            method='fir',
            iir_params=None,
            phase='zero',
            fir_window='hamming',
            fir_design='firwin',  # 'firwin2'
            pad='reflect_limited',
            verbose=self.args['verbose'],
        )

        return raw

    def save_raw(self, raw):

        fname = "{}_eeg.fif.gz".format(self.args['id'])
        eeg_channel_indices = mne.pick_types(raw.info, eeg=True)

        raw.save(fname,
                 picks=eeg_channel_indices,
                 split_size='2GB',
                 fmt='single',  # single precision is accurate enough, and backwards-compatible
                 overwrite=True,
                 split_naming='bids',  # split files given *-01, *-02
                 )

    def save_epochs(self, epochs, lab=''):

        lab = '_' + lab if lab else ''
        fname = "{}_eeg{}-epo.fif.gz".format(self.args['id'], lab)

        epochs.save(fname,
                    split_size='2GB',
                    fmt='single',  # single precision is accurate enough, and backwards-compatible
                    overwrite=True,
                    verbose=self.args['verbose'])

    def clean_data(self):

        eog = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
        bdf_labels = [self.args['id'] if self.args['id'] else [1, 2, 3, 5]]

        for i in bdf_labels:

            ### Read data ### 

            raw_fname = paths['data'].format(self.args['id'])
            raw = read_raw_bdf(
                raw_fname,
                preload=True,
                eog=eog,
                stim_channel='Status',
                verbose=self.args['verbose'])

            # get some basic visual info about the data:

            print('Data type: {}\n\n{}\n'.format(type(raw), raw))
            print('Sample rate:', raw.info['sfreq'], 'Hz')
            print('Size of the matrix: {}\n'.format(raw.get_data().shape))
            print(raw.info)
            print('\n\n {}\n'.format(raw.get_data()))

            # eeg_channel_indices = mne.pick_types(raw.info, eeg=True)
            # eog_channel_indices = mne.pick_types(raw.info, eog=True)
            # stim_channel_indices = mne.pick_types(raw.info, stim=True)

            # get stimuli timings:

            timings = paths['timings'].format(self.args['id'])
            timings = pd.read_csv(timings, encoding='utf16', skiprows=1, sep='\t')
            # timings = timings.loc[:, timings.columns.values[32:]]

            ### Events ###

            fixation_onset = timings['fixation.OnsetTime']
            stimulus_onset = timings['stimulus.OnsetTime']
            rest_onset = timings['rest.OnsetTime']
            trigger_id = timings['imgTrigger']
            displayed_word = timings['InnerWord']
            semantic_class = timings['Condition']

            event_dict = {i: j for i, j in set(zip(trigger_id, displayed_word))}
            # event_dict[1] = 'fixation' # REMOVE, UNNEEDED
            # event_dict[2] = 'rest' # REMOVE, UNNEEDED
            event_id = {i: j for j, i in event_dict.items()}  # needed for epoching later
            events = []
            # Fix instances where there are anomolous preceding trigger events: 
            for i in range(len(stimulus_onset)):
                events.append(np.array([fixation_onset[i], 0, 1]))
                events.append(np.array([stimulus_onset[i], 0, trigger_id[i]]))
                events.append(np.array([rest_onset[i], 0, 2]))
            events = np.array(events)
            _events = mne.find_events(raw, stim_channel='Status')
            events = _events[np.where(_events[:, 2] == events[1][2])[0][0] - 1:]

            ### Values for epochs (later) - determined by experimental paradigm + length of events ###
            tmin = 0
            frame = 1 / raw.info['sfreq']
            tmax = -frame + 2  # 2 seconds - removes an extra frame we don't want

            ### Annotations ###

            annot_from_events = mne.annotations_from_events(
                events=events, event_desc=event_dict, sfreq=raw.info['sfreq'])

            raw.set_annotations(annot_from_events)  # assign annotations

            ### Basic info processing ###

            # all externals apparently used (EXG7 and EXG8 recorded junk)
            raw.drop_channels(['EXG7', 'EXG8'])  # Unused; junk channels.

            # Sanity-check via STIM channel:
            # stim = raw.copy().pick(['Status']).load_data() # copy, otherwise modifies in place
            # data, times = stim[:]

            biosemi_montage = mne.channels.make_standard_montage('biosemi64')
            raw.set_montage(biosemi_montage)

            notches = np.arange(50, 100, 150)
            ref_channels = ['EXG1', 'EXG2']

            raw = self.re_reference(raw, ref_channels)
            raw = self.remove_baseline_drift(raw)  # Filter the data to remove low-frequency drifts (i.e. DC drifts)
            raw = self.remove_line_noise(raw, notches)

            raw_copy = raw.copy()

            epochs = self.make_epochs_from_sliding_window(raw_copy)
            interpolated_channels, epochs = self.interpolate_bad_channels(
                epochs)  # Before we do anything, we're now going to see if we have any bad channels using ransac
            epochs = self.re_reference(epochs,
                                       ref_channels)  # Re-reference the data, following possible intopolation of bad channels
            epochs = self.reject_bad_epochs(
                epochs)  # run autoreject (local) and supply the bad epochs detected by it to the ICA algorithm for a robust fit

            ica = self.fit_ICA(epochs, interpolated_channels=interpolated_channels)

            # Now we have a robust ICA fit:

            ica.apply(raw)  # Now we apply the ICA fit onto our original raw data

            # interpolated_channels, epochs = self.interpolate_bad_channels(epochs)
            # if interpolated_channels:
            #    epochs = self.re_reference(epochs,ref_channels) # Re-reference the data, following possible intopolation of bad channels

            if self.args['save']:
                self.save_raw(raw)

            epochs = self.make_epochs_from_events(raw, events, event_id, tmin, tmax)
            self.save_epochs(epochs)

            for band, fmin, fmax in self.args['freqs']:
                raw = self.subband_decomposition(fmin, fmax)
                epochs = self.make_epochs_from_events(raw, events, event_id, tmin, tmax)
                self.save_epochs(epochs, lab=band)

    def main(self):

        if self.args['freqs']:
            fnames = ["{}_eeg_{}-epo.fif.gz".format(self.args['id'], lab[0]) for lab in self.args['freqs']]
        else:
            fnames = ["{}_eeg-epo.fif.gz".format(self.args['id'])]

        if self.args['load']:
            for fname in fnames:
                if not os.path.exists(fname):
                    print(f"Cannot load data: {fname} not found. Proceeding to preprocess the data...")
                    self.clean_data()

        X_dat = []

        if self.args['stats']:
            for i in fnames:
                X, y = super().main(i)
                X_dat.append(X)
            X = np.hstack(X_dat)
            X = np.array(X)

        else:
            for i in fnames:
                epoch = mne.read_epochs(i, preload=True, verbose=self.args['verbose'])
                X = epoch.get_data()
                X = X.reshape(X.shape[0], -1)
                y = np.array([i for i in epoch._metadata['event_name']])
                X_dat.append(X)
            X = np.hstack(X_dat)
            X = np.array(X)

        return X, y


class RandomForest:

    def __init__(self, _):
        self.args = _

    def main(self, X, y):
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import AdaBoostClassifier

        clf = AdaBoostClassifier(n_estimators=100)
        scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1, verbose=0)
        print(f"AdaBoost = {scores}. Mean accuracy = {scores.mean()}")


def get_fMRI(mode=None):
    import nibabel as nib
    from sklearn.decomposition import PCA
    pca = PCA(1)  # super vicious - just want to keep 10 PCs

    """very rough-and-ready"""

    IspeecAI = os.path.join(os.path.expanduser("~"), 'IspeecAI')
    IspeecAI_fMRI = os.path.join(IspeecAI, 'fMRI-rec')
    IspeecAI_fMRIproc = os.path.join(IspeecAI, 'fMRI-proc')
    betas_part1 = os.path.join(IspeecAI_fMRI, 'betas', 'data', 'beta_sub1_sess1')
    betas_part2 = os.path.join(IspeecAI_fMRI, 'betas', 'data', 'beta_sub1_sess2')
    beta_labs_part1 = os.path.join(IspeecAI_fMRI, 'betas', 'label_info', 'beta_labels_subject1_session1.txt')
    beta_labs_part2 = os.path.join(IspeecAI_fMRI, 'betas', 'label_info', 'beta_labels_subject1_session2.txt')
    unsmoothed_part1 = os.path.join(IspeecAI_fMRI, 'unsmoothed', 'auCMRR_sub01_sess01.nii')
    unsmoothed_part2 = os.path.join(IspeecAI_fMRI, 'unsmoothed', 'auCMRR_sub01_sess02.nii')
    unsmoothed_labs_part1 = os.path.join(IspeecAI_fMRI, 'unsmoothed', 'sub1_session1_timings.txt')
    unsmoothed_labs_part2 = os.path.join(IspeecAI_fMRI, 'unsmoothed', 'sub1_session2_timings.txt')

    if mode == 'beta':

        with open(beta_labs_part1, 'r') as f1:
            with open(beta_labs_part2, 'r') as f2:
                labs1 = [re.sub(r'[0-9]+', '', i.split()[0]) for i in f1.readlines()]
                labs2 = [re.sub(r'[0-9]+', '', i.split()[0]) for i in f2.readlines()]
                y = np.array(labs1 + labs2)
        fnames = [os.path.join(betas_part1, i) for i in os.listdir(betas_part1)] + [os.path.join(betas_part2, i) for i
                                                                                    in os.listdir(betas_part2)]

        X = []

        for fname in fnames:
            img = np.nan_to_num(nib.load(fname).get_fdata())
            img = img.reshape(img.shape[0], -1)
            img = pca.fit_transform(img).flatten()
            X.append(img)

        X = np.array(X)

    elif mode == 'unsmoothed':

        with open(unsmoothed_labs_part1, 'r') as f1:
            with open(unsmoothed_labs_part2, 'r') as f2:
                lines1 = f1.readlines()
                labs1 = [i.split()[0] for i in lines1[1:]]
                idx1 = [(int(eval(i.split()[2])), int(eval(i.split()[2]) + 2)) for i in lines1[1:]]
                lines2 = f2.readlines()
                labs2 = [i.split()[0] for i in lines2[1:]]
                idx2 = [(int(eval(i.split()[2])), int(eval(i.split()[2]) + 2)) for i in lines2[1:]]
                y = np.array(labs1 + labs2)

        X = []

        _img = nib.load(unsmoothed_part1).get_fdata()

        for i in idx1:
            img = _img[:, :, :, i[0]:i[1]]
            img = img.reshape(img.shape[0], -1)
            img = pca.fit_transform(img).flatten()
            X.append(img)

        _img = nib.load(unsmoothed_part2).get_fdata()

        for i in idx2:
            img = _img[:, :, :, i[0]:i[1]]
            img = img.reshape(img.shape[0], -1)
            img = pca.fit_transform(img).flatten()
            X.append(img)

        X = np.array(X)

    else:

        print('no fMRI mode selected')
        exit()

    return X, y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-t", "--train", action="store_true", help="train network from EEG data")
    mode.add_argument("-d", "--test", action="store_true", help="use network to decode test EEG data")
    mode.add_argument("-w", "--debug", action="store_true", help="run in debug mode")
    parser.add_argument("-p", "--plot", action="store_true", help="plot visuals from trained network")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="run with verbose processing")
    parser.add_argument("--load", action="store_true", help="load processed data previously saved to disk")
    parser.add_argument("--save", action="store_true", help="save processed eeg data as fif.gz files")
    # ============model architecture choice=============
    # parser.add_argument("-m", "--model",type=str,choices=Models().models,required=True,
    #     help="Model architecture from one of: {}".format(', '.join(Models().models)))
    # ========thinker-(in)dependent model choice========
    parser.add_argument("--id", type=int, choices=[1, 2, 3, 5],
                        help="Subject ID number to train thinker-dependent model (default = thinker-independent model)")
    parser.add_argument("--freqs", type=ast.literal_eval,
                        default=['delta', 'theta', 'alpha', 'beta', 'gamma', 'highgamma'],
                        help="List written as \"['delta','theta','alpha','beta','gamma','highgamma']\". Delete as appropriate. \"[]\" = no subband decomposition.")
    parser.add_argument("--stats", action="store_true",
                        help="Decompose epoched data into statisitical coefficients.")
    parser.add_argument("--picks", type=str, choices=['eeg', 'brocas'], default='eeg',
                        help="Select channels for epoching and preprocessing (Nb. all channels are employed up to epoching for the purposes of ICA.")
    parser.add_argument("--fusion", action="store_true",
                        help="Fuse the EEG and fMRI data together.")
    parser.add_argument("--seed", type=int, default=99)
    # ==================================================
    args = parser.parse_args()

    picks = {
        'eeg': mne.channels.make_standard_montage('biosemi64').ch_names,
        'brocas': ['F5', 'FC5', 'C5', 'CP5', 'P5', 'FC3', 'C3', 'CP3'],
    }

    freqs = {
        'delta': (1, 3),
        'theta': (4, 7),
        'alpha': (8, 12),
        'beta': (13, 29),
        'gamma': (30, 99),
        'highgamma': (100, 255)
    }

    args.freqs = [(i, freqs[i][0], freqs[i][1]) for i in args.freqs]
    args.picks = picks[args.picks]

    # ============hardcoded data parameters=============
    test_size = 0.2  # of all data, percentage for test set
    train_size = 0.75  # of the training data, the train/val split
    # which gives us train/test/val of 6/2/2

    # =============softcoded default paths==============
    cwd = os.getcwd()
    IspeecAI = os.path.join(os.path.expanduser("~"), 'IspeecAI')
    IspeecAI_EEG = os.path.join(IspeecAI, 'EEG-rec')
    IspeecAI_fMRI = os.path.join(IspeecAI, 'fMRI-rec')
    IspeecAI_EEGproc = os.path.join(IspeecAI, 'EEG-proc')
    IspeecAI_fMRIproc = os.path.join(IspeecAI, 'EEG-proc')
    _a = os.path.join(IspeecAI_EEG, 'InnerSpeech-EEG-0014.es3')  # E-Prime experiment file
    _b = os.path.join(IspeecAI_EEG, 'InnerSpeech-EEG-0014.wndpos')  # E-Studio open windows & positions file
    _c = os.path.join(IspeecAI_EEG, 'InnerSpeech-EEG-0014-0{}-1.edat3')  # E-DataAid experiment data file
    _d = os.path.join(IspeecAI_EEG, 'InnerSpeech-EEG-0014-0{}-1.txt')  # E-Studio newline-separated timings file
    _e = os.path.join(IspeecAI_EEG, 'InnerSpeech-EEG-0014-0{}-1-ExperimentAdvisorReport.xml')  # E-Prime advisor file
    timings = os.path.join(IspeecAI_EEG, 'InnerSpeech-EEG-0014-0{}-1-export.txt')  # tab-separated timings file
    data = os.path.join(IspeecAI_EEG, 'subject0{}_session01.bdf')  # BioSemi signal data file

    os.chdir(IspeecAI)  # jump to correct directory
    print(f"Jumping to working directory: {os.getcwd()}")
    paths = {
        'IspeecAI': IspeecAI,
        'IspeecAI_EEG': IspeecAI_EEG,
        'IspeecAI_fMRI': IspeecAI_EEGproc,
        'IspeecAI_EEGproc': IspeecAI_fMRI,
        'IspeecAI_fMRIproc': IspeecAI_fMRIproc,
        '_a': _a,
        '_b': _b,
        '_c': _c,
        '_d': _d,
        '_e': _e,
        'timings': timings,
        'data': data,
    }

    # ================preprocess the data=================

    pp = Preprocess(vars(args), paths)
    sd = StatisticalDecomposition(vars(args))
    rf = RandomForest(vars(args))

    X, y = pp.main()

    if args.fusion:
        _X, _y = get_fMRI(mode='beta')
        idx_list = []
        enum = [i for i in enumerate(_y)]
        random.seed(123)
        random.shuffle(enum)
        for i in y:
            idx = [i[1] for i in enum].index(i)
            idx_list.append(enum.pop(idx)[0])
        _X = _X[idx_list]  # take re-ordered fMRI
        X = np.hstack((X, _X))  # Concat data

    rf.main(X, y)
