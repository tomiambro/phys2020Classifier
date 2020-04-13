#!/usr/bin/env python

import numpy as np
from scipy.signal import butter, lfilter
from scipy import stats
from scipy import fft

import neurokit2 as nk
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

def detect_peaks(ecg_measurements,signal_frequency,gain):

        """
        Method responsible for extracting peaks from loaded ECG measurements data through measurements processing.

        This implementation of a QRS Complex Detector is by no means a certified medical tool and should not be used in health monitoring. 
        It was created and used for experimental purposes in psychophysiology and psychology.
        You can find more information in module documentation:
        https://github.com/c-labpl/qrs_detector
        If you use these modules in a research project, please consider citing it:
        https://zenodo.org/record/583770
        If you use these modules in any other project, please refer to MIT open-source license.

        If you have any question on the implementation, please refer to:

        Michal Sznajder (Jagiellonian University) - technical contact (msznajder@gmail.com)
        Marta lukowska (Jagiellonian University)
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/c-labpl/qrs_detector
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        
        MIT License
        Copyright (c) 2017 Michal Sznajder, Marta Lukowska
    
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

        """


        filter_lowcut = 0.001
        filter_highcut = 15.0
        filter_order = 1
        integration_window = 30  # Change proportionally when adjusting frequency (in samples).
        findpeaks_limit = 0.35
        findpeaks_spacing = 100  # Change proportionally when adjusting frequency (in samples).
        refractory_period = 240  # Change proportionally when adjusting frequency (in samples).
        qrs_peak_filtering_factor = 0.125
        noise_peak_filtering_factor = 0.125
        qrs_noise_diff_weight = 0.25


        # Detection results.
        qrs_peaks_indices = np.array([], dtype=int)
        noise_peaks_indices = np.array([], dtype=int)


        # Measurements filtering - 0-15 Hz band pass filter.
        filtered_ecg_measurements = bandpass_filter(ecg_measurements, lowcut=filter_lowcut, highcut=filter_highcut, signal_freq=signal_frequency, filter_order=filter_order)

        filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]

        # Derivative - provides QRS slope information.
        differentiated_ecg_measurements = np.ediff1d(filtered_ecg_measurements)

        # Squaring - intensifies values received in derivative.
        squared_ecg_measurements = differentiated_ecg_measurements ** 2

        # Moving-window integration.
        integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window)/integration_window)

        # Fiducial mark - peak detection on integrated measurements.
        detected_peaks_indices = findpeaks(data=integrated_ecg_measurements,
                                                     limit=findpeaks_limit,
                                                     spacing=findpeaks_spacing)

        detected_peaks_values = integrated_ecg_measurements[detected_peaks_indices]

        return detected_peaks_values,detected_peaks_indices

 
def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
        """
        Method responsible for creating and applying Butterworth filter.
        :param deque data: raw data
        :param float lowcut: filter lowcut frequency value
        :param float highcut: filter highcut frequency value
        :param int signal_freq: signal frequency in samples per second (Hz)
        :param int filter_order: filter order
        :return array: filtered data
        """
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

def findpeaks(data, spacing=1, limit=None):
        """
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        Finds peaks in `data` which are of `spacing` width and >=`limit`.
        :param ndarray data: data
        :param float spacing: minimum spacing to the next peak (should be 1 or more)
        :param float limit: peaks should have value greater or equal
        :return array: detected peaks indexes array
        """
        len = data.size
        x = np.zeros(len + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len] = data
        peak_candidate = np.zeros(len)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before
            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
        if limit is not None:
            ind = ind[data[ind] > limit]
        return ind


def get_12ECG_features(data, header_data):

    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs= int(tmp_hea[2])
    gain_lead = np.zeros(num_leads)
    
    for ii in range(num_leads):
        tmp_hea = header_data[ii+1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # for testing, we included the mean age of 57 if the age is a NaN
    # This value will change as more data is being released
    for iline in header_data:
        if iline.startswith('#Age'):
            tmp_age = iline.split(': ')[1].strip()
            age = int(tmp_age if tmp_age != 'NaN' else 57)
        elif iline.startswith('#Sex'):
            tmp_sex = iline.split(': ')[1]
            if tmp_sex.strip()=='Female':
                sex =1
            else:
                sex=0
        elif iline.startswith('#Dx'):
            label = iline.split(': ')[1].split(',')[0]


#   We are only using data from lead1
    peaks,idx = detect_peaks(data[0],sample_Fs,gain_lead[0])
   
#   mean
    mean_RR = np.mean(idx/sample_Fs*1000)
    mean_Peaks = np.mean(peaks*gain_lead[0])

#   median
    median_RR = np.median(idx/sample_Fs*1000)
    median_Peaks = np.median(peaks*gain_lead[0])

#   standard deviation
    std_RR = np.std(idx/sample_Fs*1000)
    std_Peaks = np.std(peaks*gain_lead[0])

#   variance
    var_RR = stats.tvar(idx/sample_Fs*1000)
    var_Peaks = stats.tvar(peaks*gain_lead[0])

#   Skewness
    skew_RR = stats.skew(idx/sample_Fs*1000)
    skew_Peaks = stats.skew(peaks*gain_lead[0])

#   Kurtosis
    kurt_RR = stats.kurtosis(idx/sample_Fs*1000)
    kurt_Peaks = stats.kurtosis(peaks*gain_lead[0])

    features = np.hstack([age,sex,mean_RR,mean_Peaks,median_RR,median_Peaks,std_RR,std_Peaks,var_RR,var_Peaks,skew_RR,skew_Peaks,kurt_RR,kurt_Peaks])

  
    return features


def get_12ECG_features_labels(data, header_data):

    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs= int(tmp_hea[2])
    gain_lead = np.zeros(num_leads)
    
    for ii in range(num_leads):
        tmp_hea = header_data[ii+1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # for testing, we included the mean age of 57 if the age is a NaN
    # This value will change as more data is being released
    for iline in header_data:
        if iline.startswith('#Age'):
            tmp_age = iline.split(': ')[1].strip()
            age = int(tmp_age if tmp_age != 'NaN' else 57)
        elif iline.startswith('#Sex'):
            tmp_sex = iline.split(': ')[1]
            if tmp_sex.strip()=='Female':
                sex =1
            else:
                sex=0
        elif iline.startswith('#Dx'):
            label = iline.split(': ')[1].split(',')[0]


    N = len(data[0])
    sp= sample_Fs/N    # resolución espectral

    Y = np.fft.fft(data[0])
    ff = np.linspace(0, (N/2)*sp N/2).flatten()
    fmax = float(ff[np.where(np.abs(Y[0:N//2]) == max(np.abs(Y[0:N//2])))])


#   We are only using data from lead1
    peaks,idx = detect_peaks(data[0],sample_Fs,gain_lead[0])
       
#   mean
    mean_RR = np.mean(idx/sample_Fs*1000)
    mean_R_Peaks = np.mean(peaks*gain_lead[0])

#   median
    median_RR = np.median(idx/sample_Fs*1000)
    median_R_Peaks = np.median(peaks*gain_lead[0])

#   standard deviation
    std_RR = np.std(idx/sample_Fs*1000)
    std_R_Peaks = np.std(peaks*gain_lead[0])

#   variance
    var_RR = stats.tvar(idx/sample_Fs*1000)
    var_R_Peaks = stats.tvar(peaks*gain_lead[0])

#   Skewness
    skew_RR = stats.skew(idx/sample_Fs*1000)
    skew_R_Peaks = stats.skew(peaks*gain_lead[0])

#   Kurtosis
    kurt_RR = stats.kurtosis(idx/sample_Fs*1000)
    kurt_R_Peaks = stats.kurtosis(peaks*gain_lead[0])

#   RMSSD (HRV)
    rmssd = np.sqrt(np.mean(np.square(np.diff(idx))))

#   All Peaks
    ecg_signal = nk.ecg_clean(data[0], sampling_rate=sample_Fs, method="biosppy")
    _ , rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sample_Fs)
    try:
        signal_peak, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sample_Fs)
        t_peaks = waves_peak['ECG_T_Peaks']
        p_peaks = waves_peak['ECG_P_Peaks']
        q_peaks = waves_peak['ECG_Q_Peaks']
        s_peaks = waves_peak['ECG_S_Peaks']
        p_onsets = waves_peak['ECG_P_Onsets']
        t_offsets = waves_peak['ECG_T_Offsets']
    except ValueError:
        print('Exception raised!')
        pass

#   T Peaks
    t_peaks = np.asarray(t_peaks, dtype=float)
    t_peaks = t_peaks[~np.isnan(t_peaks)]
    t_peaks = [int(a) for a in t_peaks]
    mean_T_Peaks = np.mean([data[0][w] for w in t_peaks])

#   P peaks
    p_peaks = np.asarray(p_peaks, dtype=float)
    p_peaks = p_peaks[~np.isnan(p_peaks)]
    p_peaks = [int(a) for a in p_peaks]
    mean_P_Peaks = np.mean([data[0][w] for w in p_peaks])    

#   Q peaks
    q_peaks = np.asarray(q_peaks, dtype=float)
    q_peaks = q_peaks[~np.isnan(q_peaks)]
    q_peaks = [int(a) for a in q_peaks]
    mean_Q_Peaks = np.mean([data[0][w] for w in q_peaks])

#   S peaks
    s_peaks = np.asarray(s_peaks, dtype=float)
    s_peaks = s_peaks[~np.isnan(s_peaks)]
    s_peaks = [int(a) for a in s_peaks]
    mean_S_Peaks = np.mean([data[0][w] for w in s_peaks])

#   P Onsets
    p_onsets = np.asarray(p_onsets, dtype=float)
    # p_onsets = p_onsets[~np.isnan(p_onsets)]
    mean_P_Onsets = np.mean(p_onsets/sample_Fs*1000)

#   T Onsets
    t_offsets = np.asarray(t_offsets, dtype=float)
    # t_offsets = t_offsets[~np.isnan(t_offsets)]
    mean_T_offsets = np.mean(t_offsets/sample_Fs*1000)

    features = [age,sex,fmax,mean_RR,mean_R_Peaks,mean_T_Peaks,mean_P_Peaks,mean_Q_Peaks,mean_S_Peaks,median_RR,median_R_Peaks,std_RR,std_R_Peaks,var_RR,var_R_Peaks,skew_RR,skew_R_Peaks,kurt_RR,kurt_R_Peaks,mean_P_Onsets,mean_T_offsets,rmssd,label]
  
    return features