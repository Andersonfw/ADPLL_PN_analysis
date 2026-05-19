import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import locale
import datetime
import os
import scienceplots as sp
import pandas as pd

import function as fn

class Colors:
    WHITE = '\033[97m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    YELLOW = '\033[0;33m'
    BLUE = "\033[0;34m"

'''------------------------------------------------------------------------------------------------------- '''

# --- load Files ---
#ntw_file = np.loadtxt("data/tune_ntw.txt")
# fsm_file = pd.read_csv("data/fsm_states_old.csv", sep=';', header=None)
fsm_file = pd.read_csv("data/fsm_states.csv", sep=';', header=None)
bank_files = pd.read_csv("data/bank_cap.csv", sep=';', header=None)
fout = pd.read_csv("data/fout.csv", sep=';', header=None)
t_edges = np.loadtxt("data/close_loop_edge_times.txt")


# N_edges = len(t_edges)
# # Frequência média (portadora)
# f_c = (N_edges - 1) / (t_edges[-1] - t_edges[0])

# print(f"Frequência média da portadora (f_c): {f_c/1e9:.6f} GHz")
# fout = []

# for i in range(0, N_edges-1):
#     fout.append((1/(t_edges[i+1] - t_edges[i])))

# fout.append(fout[-1])
# fout = np.array(fout)

# t_periods = np.diff(t_edges)
# f_inst = 1.0 / t_periods

# # Configuração da Média Móvel
# window_size = 100  # Tamanho da janela (ex: 1000 ciclos)
# window = np.ones(window_size) / window_size

# # Cálculo da média móvel
# f_smooth = np.convolve(f_inst, window, mode='same')

# # Para medir o Drift (Desvio em relação à média)
# drift = f_smooth - f_c

# fout = np.array(f_smooth)
# fout = fout[1000:-10000]  # Remove o primeiro e último valor, que são afetados pela média móvel
# print(f"Max Drift observado (suavizado): {np.max(np.abs(drift))/1e3:.2f} kHz")
# plt.style.use(['science','ieee'])
# plt.style.use(['science','ieee'])
# plt.rcParams['legend.frameon'] = True  # Mostrar a moldura da legenda
# plt.rcParams['legend.edgecolor'] = 'lightgray'  # Cor da borda da legenda
# plt.rcParams['legend.facecolor'] = 'lightgray'  # Cor do fundo da legenda

# plt.figure(figsize=(6,4), dpi=600)
# label1 = "Normalized Tuning Word"
# plt.plot(ntw_file[1000:], label=f"NTW")
# plt.grid(visible=True)
# plt.legend(facecolor='white', framealpha=1, fontsize=12)
# plt.xticks(fontsize=12)
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('NTW', fontsize=12)
# # plt.tight_layout()
# plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\phase_noise_calc\img\NTW.png', bbox_inches='tight', format='png')
# # plt.show()

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
# t_edges = t_edges[1000:N_edges-10001]  # Ajusta o tamanho de t_edges para corresponder ao tamanho de fout
f_required = 2.48e9 #np.mean(fout[1])  # Hz
f_req_array = np.full_like(fout, f_required)
print(f"{Colors.YELLOW} Required Frequency at the output: {f_required/1e6:.6f} MHz {Colors.RESET}")

label1 = "Frequency error at the output"
axs[0, 0].plot(fout[0] * 1e6, (fout[1] - f_required)/1e3, label=label1, color='red')
axs[0, 0].grid(visible=True)
axs[0, 0].legend(facecolor='white', framealpha=1, fontsize=12)
axs[0, 0].set_xlabel('Time (us)', fontsize=12)
axs[0, 0].set_ylabel('Frequency (kHz )', fontsize=12)
# plt.show()


label1 = "Frequency output"
axs[1, 1].plot(fout[0]  * 1e6,  (fout / 1e6), label=label1, color='red')
axs[1, 1].grid(visible=True)
axs[1, 1].legend(facecolor='white', framealpha=1, fontsize=12)
axs[1, 1].set_xlabel('Time (us)', fontsize=12)
axs[1, 1].set_ylabel('Frequency (kHz )', fontsize=12)
# plt.show()


# plt.figure(figsize=(6,4), dpi=600)
# text_labels = ['IDLE', 'START', 'STAGE1', 'STAGE2', 'STAGE3', 'SETTLE', 'KTDC_CALIB', 'KDCO_KALIB']
text_labels = ['IDLE', 'START', 'STAGE1', 'STAGE2', 'STAGE3', 'SETTLE']
# 2. Obtém os valores únicos de Y no seu dado para usar como "localizações" dos ticks
# Isso garante que o rótulo 'START' corresponda ao menor valor Y, 'STEP1' ao próximo, e assim por diante.
# Assumindo que os valores únicos em fsm_file[0] são 1, 2, 3, 4, 5 e estão em ordem crescente.
unique_y_values = sorted(fsm_file[0].unique())
axs[0, 1].set_yticks(unique_y_values, text_labels)
axs[0, 1].plot( fsm_file[1] * 1e6,fsm_file[0], label=f"FSM States", color='red')
axs[0, 1].grid(visible=True)
axs[0, 1].legend(facecolor='white', framealpha=1, fontsize=12)
axs[0, 1].set_xlabel('Time (us)', fontsize=12)
axs[0, 1].set_ylabel('STATE', fontsize=12)
# plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\phase_noise_calc\img\FSM.png', bbox_inches='tight', format='png')
# plt.show()


# plt.figure(figsize=(6,4), dpi=600)
axs[1, 0].plot(  bank_files[0] * 1e6, bank_files[1], label=f"PVT Bank", color='blue')
axs[1, 0].plot( bank_files[0] * 1e6, bank_files[2], label=f"AQ Bank", color='green')
axs[1, 0].plot( bank_files[0] * 1e6, bank_files[3], label=f"TB Bank", color='red')
# axs[1, 0].plot( bank_files[0] * 1e6, bank_files[4], label=f"TB SDM", color='purple')
# plt.plot( bank_files[3], label=f"TBS_I", color='blue')
axs[1, 0].grid(visible=True)
axs[1, 0].legend(facecolor='white', framealpha=1, fontsize=12)
axs[1, 0].set_xlabel('Time (us)', fontsize=12)
axs[1, 0].set_ylabel('Capacitor bank value', fontsize=12)
# plt.tight_layout()
# plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\phase_noise_calc\img\dco_bank.png', bbox_inches='tight', format='png')
plt.show()