import ctypes
from pathlib import Path as path
from pickle import FALSE, TRUE

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import locale
import datetime
import os
import scienceplots as sp
import pandas as pd
import scipy
from sympy import false, true

import function as fn
import utilities as ut
import process as pr
import utilities as ut


# ++++++++++++++ DEFINITIONS ++++++++++++++
IEEE_PICTURES = false
PN_ANALYSIS = TRUE

window_time = 0.5e-6  # Tamanho da janela para suavização (1us para BLE)
time_cut_PN_start = 2.5e-04
time_cut_freq_anal_start = time_cut_PN_start#5.0e-4 #1.2e-04
time_cut_freq_anal_stop = time_cut_freq_anal_start + 120e-6

FREQ = "2418123" #2402 2440 2418123 2480
CORNER = "TYP" # WORST TYP BEST
SETTING = "FPREDICT_SDM1" # FPREDICT FPREDICT_SDM1 FPREDICT_SDM2
TDC_DCO_CORNER = "TYP" # TDC DCO
# path_string = "data/"+freq
# path_string = "data/TYP/2480_WORST/FPREDICT"
path_string = "data/"+TDC_DCO_CORNER+"/"+FREQ+"_"+CORNER+"/"+SETTING
# path_string = "data/SIM_DATA" 
data_path = path(path_string)

if len(FREQ) > 4:
    freq_formatada = f"{FREQ[:4]}.{FREQ[4:]}"  
    freq_atual = float(freq_formatada)
else:
    freq_atual = float(FREQ)

f_required = freq_atual * 1e6 #2.440e9 #2.39205e9 #2.402e9 #   #np.mean(fout[1])  # Hz 2.402e9

# --- load Files ---
fsm_path = ut.get_latest_file(data_path, "fsm_states", "csv")
fsm_file = pd.read_csv(fsm_path, sep=';', header=None)
t_edges_name = "close_loop_edge_times_"

t_edges_path  = ut.get_latest_file(data_path, t_edges_name, "txt")
bank_path     = ut.get_latest_file(data_path, "bank_cap", "csv")
phe_path      = ut.get_latest_file(data_path, "phe", "csv")
otw_path      = ut.get_latest_file(data_path, "otw", "csv")
# active_settings_path = ut.get_latest_file(data_path, "sim_historic", "csv")

bank_files = pd.read_csv(bank_path, sep=';', header=None)
t_edges = np.loadtxt(t_edges_path)
phe = pd.read_csv(phe_path, sep=';', header=None)
otw = pd.read_csv(otw_path, sep=';', header=None)
# active_settings = pd.read_csv(active_settings_path, sep=';', header=None)

i_start_banks = 0 #(np.abs(bank_files[0] - time_cut_freq_anal_start)).argmin()
i_stop_banks = (np.abs(bank_files[0].values - 250e-6)).argmin()
i_stop_ckv = (np.abs(t_edges - 250e-6)).argmin()
# Corta o DataFrame INTEIRO de uma vez só
# O +1 garante que o elemento do índice i_stop_banks seja incluído, se desejado
bank_files = bank_files.iloc[i_start_banks : i_stop_banks + 1].reset_index(drop=True)
fsm_file.loc[len(fsm_file)] = [5,  bank_files[0].iloc[-1]]  # Adiciona um ponto extra para manter a linha até o final do tempo

print(f"{ut.Colors.BLUE}\r\n--------------------------------------------------------------------"
        "\r\nFILES LOADED",)
print(f"{ut.Colors.YELLOW}\r\n Path files Loaded: {data_path}")
print(f"{ut.Colors.YELLOW}\r\n Egdes_file: {t_edges_path.name}")
print(f"{ut.Colors.YELLOW}\r\n Bank_file: {bank_path.name}")
print(f"{ut.Colors.YELLOW}\r\n OTW_file: {otw_path.name}")
print(f"{ut.Colors.YELLOW}\r\n PHE_file: {phe_path.name}")
# print(f"{ut.Colors.YELLOW}\r\n ACTIVE SETTINGS: {active_settings[1][len(active_settings)-1]} From Date {active_settings[0][len(active_settings)-1]} ")
print(f"{ut.Colors.BLUE}\r\n--------------------------------------------------------------------",)


pr.process(t_edges, 
            window_time, 
            time_cut_freq_anal_start, 
            time_cut_freq_anal_stop, 
            f_required, 
            time_cut_PN_start, 
            fsm_file, 
            bank_files, 
            phe, 
            otw, 
            i_stop_ckv, 
            i_stop_banks, 
            plot_all=true, 
            IEEE_en=IEEE_PICTURES, 
            PN_analysis=PN_ANALYSIS)
