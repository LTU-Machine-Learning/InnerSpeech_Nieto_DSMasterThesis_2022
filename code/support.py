import warnings
import mne
import numpy as np
import os

from sklearn import model_selection
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

from aux.pre_process import extract_data_from_subject, select_time_window, filter_by_condition

root_dir = "./ds003626"
all_subjects = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def download_data():
    os.system('pip3 install mne -q')
    os.system('pip3 install awscli')
    os.system('aws s3 sync --no-sign-request s3://openneuro.org/ds003626 ds003626/')


def init_repo():
    np.random.seed(23)
    mne.set_log_level(verbose='warning')  # to avoid info at terminal
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore")
    warnings.simplefilter('ignore')
    download_data()


def get_subjects_data(condition, t_start=1, t_end=2.5, fs=256, subjects=all_subjects):
    data = []
    labels = []
    groups = []

    for subject in subjects:
        x, y = extract_data_from_subject(root_dir, subject, "EEG")
        x = select_time_window(X=x, t_start=t_start, t_end=t_end, fs=fs)
        x, y = filter_by_condition(x, y, condition)
        data.append(x)
        labels.append(y.T[1])
        groups.append(np.full(len(x), subject))

    data_array = np.vstack(data)
    label_array = np.hstack(labels)
    group_array = np.hstack(groups)

    return data_array, label_array, group_array


def run_cross_validation(classifier, k_fold, x, y, group):
    results = model_selection.cross_val_score(classifier, x, y, cv=k_fold, scoring='accuracy', groups=group)
    return results.mean()


def get_feature_selection_model(features_a, label_a):
    scl = StandardScaler()
    features_a = scl.fit_transform(features_a)

    lsvc = LinearSVC(C=0.01, penalty="l2", dual=False).fit(features_a, label_a)
    fsm = SelectFromModel(lsvc, prefit=True)
    return fsm


def print_report_nested_cross_validation(splits_list, params_list, scores_list):
    print("Nested cross-validation:")
    print('{:<30} {:<20} {:<15}'.format("Split", "Mean CV Score", "Best parameter"))
    for index, split in enumerate(splits_list):
        split_string = "{}/{}".format(100 - (split * 100), split * 100)
        score = "{:.3f}".format(scores_list[index])
        params = "{}".format(params_list[index])
        print('{:<30} {:<20} {:<15}'.format(split_string, score, params))


def print_report_classifier(splits_list, ac_list, cv_score_list):
    print("\nClassification:")
    print('{:<30} {:<20} {:<15}'.format("Split", "Accuracy", "Cross validation"))
    for index, split in enumerate(splits_list):
        split_string = "{}/{}".format(100 - (split * 100), split * 100)
        print('{:<30} {:<20} {:<15}'.format(split_string, ac_list[index], cv_score_list[index]))
