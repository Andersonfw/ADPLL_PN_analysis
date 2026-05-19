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
ntw_file = np.loadtxt("data/tune_ntw.txt")
fsm_file = pd.read_csv("data/fsm_states.csv", sep=';', header=None)
bank_files = pd.read_csv("data/bank_cap.csv", sep=';', header=None)
fout = pd.read_csv("data/fout.csv", sep=';', header=None)

plt.style.use(['science','ieee'])
plt.style.use(['science','ieee'])
plt.rcParams['legend.frameon'] = True  # Mostrar a moldura da legenda
plt.rcParams['legend.edgecolor'] = 'lightgray'  # Cor da borda da legenda
plt.rcParams['legend.facecolor'] = 'lightgray'  # Cor do fundo da legenda

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

f_required = np.mean(fout[1])  # Hz
print(f"{Colors.YELLOW} Required Frequency at the output: {f_required/1e6:.6f} MHz {Colors.RESET}")

plt.figure(figsize=(6,4), dpi=600)
label1 = "Frequency error at the output"
plt.plot(fout[0] * 1e6, (fout[1] - f_required)/1e3, label=label1, color='red')
plt.grid(visible=True)
plt.legend(facecolor='white', framealpha=1, fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Time (us)', fontsize=12)
plt.ylabel('Frequency (kHz )', fontsize=12)
plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\phase_noise_calc\img\freq_output.png', bbox_inches='tight', format='png')
# # plt.show()


plt.figure(figsize=(6,4), dpi=600)
text_labels = ['IDLE', 'START', 'STAGE1', 'STAGE2', 'STAGE3', 'SETTLE']
# 2. Obtém os valores únicos de Y no seu dado para usar como "localizações" dos ticks
# Isso garante que o rótulo 'START' corresponda ao menor valor Y, 'STEP1' ao próximo, e assim por diante.
# Assumindo que os valores únicos em fsm_file[0] são 1, 2, 3, 4, 5 e estão em ordem crescente.
unique_y_values = sorted(fsm_file[0].unique())
plt.yticks(unique_y_values, text_labels)
plt.plot( fsm_file[1] * 1e6,fsm_file[0], label=f"FSM States", color='red')
plt.grid(visible=True)
plt.legend(facecolor='white', framealpha=1, fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Time (us)', fontsize=12)
plt.ylabel('STATE', fontsize=12)
plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\phase_noise_calc\img\FSM.png', bbox_inches='tight', format='png')
# plt.show()


plt.figure(figsize=(6,4), dpi=600)
plt.plot(  bank_files[0] * 1e6, bank_files[1], label=f"PVT Bank", color='blue')
plt.plot( bank_files[0] * 1e6, bank_files[2], label=f"AQ Bank", color='green')
plt.plot( bank_files[0] * 1e6, bank_files[3], label=f"TB Bank", color='red')
# plt.plot( bank_files[3], label=f"TBS_I", color='blue')
plt.grid(visible=True)
plt.legend(facecolor='white', framealpha=1, fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Time (us)', fontsize=12)
plt.ylabel('Capacitor bank value', fontsize=12)
# plt.tight_layout()
plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\phase_noise_calc\img\dco_bank.png', bbox_inches='tight', format='png')
# plt.show()