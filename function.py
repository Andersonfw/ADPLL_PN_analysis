""""
@author: Ânderson Felipe Weschenfelder
"""
import numpy as np
from scipy import signal

class Colors:
    WHITE = '\033[97m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    YELLOW = '\033[0;33m'
    BLUE = "\033[0;34m"


def fun_calc_psd(x , fs=1 , rbw=100e3 , fstep=None):
    '''
    Calculate power spectral density

    INPUT arguments
    x     :  data vector
    fs    :  sample rate [Hz]
    rbw   :  resolution bandwidth [Hz]
    fstep :  FFT frequency bin separation [Hz]
    OUTPUT
    XdB	: spectrum of x [dB]
    f	: frequency vector [Hz]
    '''

    if fstep is None:
        fstep = rbw / 1.62
    len_x = len(x)
    nwin = round(fs * 1.62 / rbw)
    nfft = round(fs / fstep)
    if nwin > len_x:
        nwin = len_x
        rbw = fs * 1.62 / nwin
    num_segments = 8
    # nwin = math.floor(len(x) / num_segments)
    fftstr = (f'len(x)={len_x:.2f}, rbw={rbw / 1e3:.2f}kHz, fstep={fstep / 1e3:.2f}kHz, nfft={nfft:d}, nwin={nwin:d}')
    print(f'Calculating the PSD: {fftstr}')
    f , X = signal.welch(x , fs=fs , window=signal.windows.blackman(nwin) , nperseg=nwin , nfft=nfft ,scaling='density')
    X *= (np.sinc(f / fs)) ** 2  # correct for ZOH
    XdB = 10 * np.log10(X)
    XdB_sig = np.max(XdB)
    print(f'Signal PSD peak = {XdB_sig:.2f} dB, 10log(rbw) = {10 * np.log10(rbw):.1f}')
    return XdB , f


def edges_convert_to_freq_and_smooth(t_edges, window_time, time_cut_freq_analysis,silent=False):
    

    indice_edges_analysies = (np.abs(t_edges - time_cut_freq_analysis)).argmin()
    t_edges_cut = t_edges[indice_edges_analysies:] 
    N_edges = len(t_edges_cut)
    f_c = (N_edges - 1) / (t_edges_cut[-1] - t_edges_cut[0])

    t_periods = np.diff(t_edges_cut)
    f_inst = 1.0 / t_periods

    t_periods_full_analysis = np.diff(t_edges)
    f_inst__full_analysis = 1.0 / t_periods_full_analysis
    
    avg_period = np.mean(t_periods)
    window_size = int(1e-6 / avg_period)
    window = np.ones(window_size) / window_size
    f_smooth = np.convolve(f_inst, window, mode='valid')
    f_smooth_full_analysis = np.convolve(f_inst__full_analysis, window, mode='valid')
    drift = f_smooth - f_c
    max_drift = np.max(np.abs(drift))
    
    time_axis_smooth = t_edges_cut[window_size :] # Eixo de tempo após a convolução
    dt_us = np.diff(time_axis_smooth) * 1e6   # Intervalos em microsegundos
    df = np.diff(f_smooth)                   # Variação de frequência em Hz
    drift_rates = df / dt_us
    max_drift_rate = np.max(np.abs(drift_rates))

    if not silent:
        print(f"{Colors.GREEN}--- Análise de média movel com janela de {window_size} samples --- Window Time: {window_time * 1e6:.2f} us ---\r\n")
        print(f"{Colors.GREEN}Frequência Inicial Medida: {f_c/1e6:.4f} MHz ---\r\n")
        print(f"{Colors.GREEN}Máximo Drift Medido: {max_drift/1e3:.2f} kHz  (Limite: ±50 kHz) ---\r\n")
        print(f"{Colors.GREEN}Máximo Drift Rate: {max_drift_rate:.2f} Hz/us (Limite: 400 Hz/us) ---\r\n")
        

    return {
        "f_c": f_c,
        "f_smooth": f_smooth,
        "drift": drift,
        "f_inst": f_inst,
        "t_periods": t_periods,
        "max_drift": max_drift,
        "max_drift_rate": max_drift_rate,
        "index_cut_freq_analysis": indice_edges_analysies,
        "t_edges_cut": t_edges_cut,
        "f_smooth_full_analysis": f_smooth_full_analysis,
    }


