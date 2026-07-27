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

SIM_MODE = 9
"""
    SIM_MODE = 0: Typical Corner
    SIM_MODE = 1: Slow Corner
    SIM_MODE = 2: Fast Corner
    SIM_MODE = 3: FUTURE USE
    SIM_MODE = 4: ALL OFF

    SIM_MODE = 5: FREF
    SIM_MODE = 6: SDM 1 Ordem
    SIM_MODE = 7: SDM 2 Ordem 
    SIM_MODE = x: DEFAULT SIMULATION
"""



window_time = 0.5e-6  # Tamanho da janela para suavização (1us para BLE)
time_cut_PN_start = 2.5e-04
time_cut_freq_anal_start = time_cut_PN_start#5.0e-4 #1.2e-04
time_cut_freq_anal_stop = time_cut_freq_anal_start + 120e-6

FREQ = "2418123" #2402 2440 2418123 2480
CORNER = "TYP" # WORST TYP BEST
SETTING = "FPREDICT" # FPREDICT FPREDICT_SDM1 FPREDICT_SDM2
# path_string = "data/"+freq
# path_string = "data/TYP/2480_WORST/FPREDICT"
path_string = "data/TYP/"+FREQ+"_"+CORNER+"/"+SETTING
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






if  SIM_MODE == 0:
    t_edges_path = ut.get_latest_file(data_path, "close_loop_edge_times_typ", "txt")  
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 1:
    t_edges_path = ut.get_latest_file(data_path, "close_loop_edge_times_worst", "txt")
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 2:
    t_edges_path = ut.get_latest_file(data_path, "close_loop_edge_times_best", "txt")
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 4:
    # bank_path    = ut.get_latest_file(data_path+"/ALL_OFF", "bank_cap", "csv")
    # t_edges_path = ut.get_latest_file(data_path+"/ALL_OFF", "close_loop_edge_times_typ", "txt")
    t_edges_name = "close_loop_edge_times_typ"
    data_path = path("data/"+FREQ+"/ALL_OFF")
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 5:
    # bank_path    = ut.get_latest_file(data_path, "bank_cap_SDM_off", "csv")
    # t_edges_path = ut.get_latest_file(data_path, "close_loop_edge_times_typ_SDM_off", "txt")
    t_edges_name = "close_loop_edge_times_typ"
    data_path = path("data/"+FREQ+"/FREF")
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 6:
    # bank_path =    ut.get_latest_file(data_path, "bank_cap_SDM_1_en", "csv")
    # t_edges_path = ut.get_latest_file(data_path, "close_loop_edge_times_typ_SDM_1_en", "txt")
    t_edges_name = "close_loop_edge_times_typ"
    data_path = path("data/"+FREQ+"/SDM1")
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 7:
    # bank_path =   ut.get_latest_file(data_path, "bank_cap_SDM_2_en", "csv")
    # t_edges_path = ut.get_latest_file(data_path, "close_loop_edge_times_typ_SDM_2_en", "txt")
    t_edges_name = "close_loop_edge_times_typ"
    data_path = path("data/"+FREQ+"/SDM2")
#-----------------------------------------------------------------------------------------------#
else:
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
