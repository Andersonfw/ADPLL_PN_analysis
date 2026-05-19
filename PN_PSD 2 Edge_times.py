import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import locale
import datetime
import os
import scienceplots as sp

import function as fn

class Colors:
    WHITE = '\033[97m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    YELLOW = '\033[0;33m'
    BLUE = "\033[0;34m"

'''------------------------------------------------------------------------------------------------------- '''

# --- 2. Carregar os Tempos de Borda ---
# t_edges = np.loadtxt("data/Open_loop_edge_times.txt")
t_edges = np.loadtxt("data/close_loop_edge_times.txt")
print(f"Carregados {len(t_edges)} tempos de borda.")

# --- 3. Calcular Frequência Média e Jitter ---
time_cut = 1.2e-04
indice_edges = (np.abs(t_edges - time_cut)).argmin()
t_edges = t_edges[indice_edges:]  # Cortar os tempos de borda

N_edges = len(t_edges)
# Frequência média (portadora)
f_c = (N_edges - 1) / (t_edges[-1] - t_edges[0])
# Período médio

print(f"Frequência média da portadora (f_c): {f_c/1e9:.6f} GHz")





# F_d = 4.8e9  # Frequência desejada em Hz
T_c = 1.0 / f_c

# Criar tempos de borda ideais, começando do primeiro tempo real
t_ideal = T_c + np.arange(N_edges) * T_c

# Calcular o Jitter (erro de tempo acumulado)
jitter = t_edges - t_ideal

# --- 4. Calcular a PSD do Jitter (Sj) ---
fs = f_c
print(f"{Colors.BLUE}\r\n--------------------------------------------------------------------",
        "\r\nCALCULATE THE PHASE NOISE OF DCO OUTPUT",
        "\r\n--------------------------------------------------------------------")

phase = jitter * 2 * np.pi * f_c
# H(z) = 1 - z^-1  (Um diferenciador/filtro high-pass)
# b = np.array([1, -1])
# a = np.array([1])
# x = signal.lfilter(b, a, phase - np.mean(phase))

# Xdb_o é o DSB em dB (resultado da sua função)
Xdb_o , f = fn.fun_calc_psd(phase , fs, 50e3 , 500)  

# CONVERSÂO de DSB par SSB (SSB em dB) subtraindo 3dB
Xdb_o_SSB = Xdb_o - 10 * np.log10(2)
marker = 1e6  # Substitua pelo valor específico de frequência desejado
indice = (np.abs(f - marker)).argmin()
marker_dB = Xdb_o_SSB[indice]

marker_tdc = 1e3  # Substitua pelo valor específico de frequência desejado
indice_tdc = (np.abs(f - marker_tdc)).argmin()
marker_dB_tdc = Xdb_o_SSB[indice_tdc]

    # plt.style.use(['science','ieee'])
    # plt.style.use(['science','ieee'])
    # plt.rcParams['legend.frameon'] = True  # Mostrar a moldura da legenda
    # plt.rcParams['legend.edgecolor'] = 'lightgray'  # Cor da borda da legenda
    # plt.rcParams['legend.facecolor'] = 'lightgray'  # Cor do fundo da legenda
    # # plt.figure(figsize=(3.54,2), dpi=600)
    # plt.figure(figsize=(6,4), dpi=600)
    # # plt.figure()
    # # Plotar a resposta ao degrau
    # #plt.plot(t, y)
    # for sys in range(len(pa)):
    #     qsi = 0.5 * a / (np.sqrt(pa[sys]))
    #     HOL = (a + pa[sys] * fr / s) * (fr / s)
    #     mag, phase, f, Hol = h_to_bode(H=HOL, freq=w, irr=None, margem=margem, prints=True, plot=False)
    #     Hcl_TDC = (Hol / (1 + Hol))
    #     t, y = control.step_response(Hcl_TDC)
    #     plt.plot(t, y, label=f"$\zeta$: {qsi:.2f}  $K_I$={pas[sys]}")
    # #plt.title('Resposta ao Degrau da função $H_{TDC}$ variando $\zeta$ com $K_p=2^-5$ e $f_{REF}=$ 26MHz')
    # # plt.xlabel('Time (s)')#, fontsize=12)
    # plt.xlabel('Tempo (s)', fontsize=12)
    # # plt.ylabel('Step responde')#,fontsize=12)
    # plt.ylabel('Resposta',fontsize=12)
    # plt.legend(facecolor='white', framealpha=1,fontsize=12)
    # plt.xlim(0, 2e-5)
    # plt.yticks(fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.grid()
    # plt.savefig(r'C:\Users\ander\OneDrive\Área de Trabalho\Imagens\step_htdc.eps', bbox_inches='tight', format='eps')

    # plt.tight_layout()
    # plt.show()

# plt.figure()
# plt.style.use(['science','ieee'])
# plt.style.use(['science','ieee'])
# plt.rcParams['legend.frameon'] = True  # Mostrar a moldura da legenda
# plt.rcParams['legend.edgecolor'] = 'lightgray'  # Cor da borda da legenda
# plt.rcParams['legend.facecolor'] = 'lightgray'  # Cor do fundo da legenda
# plt.figure(figsize=(3.54,2), dpi=600)
# plt.figure(figsize=(6,4), dpi=600)
label1 = "Phase Noise"
plt.semilogx(f , Xdb_o_SSB, label=label1)
plt.scatter(marker, marker_dB, color='blue', marker='o', label=f'{marker_dB:.2f} dBc/Hz  @1 MHz')
plt.scatter(marker_tdc, marker_dB_tdc, color='red', marker='o', label=f'{marker_dB_tdc:.2f} dBc/Hz  @100 kHz')
plt.grid(visible=True)
plt.legend(facecolor='white', framealpha=1, fontsize=12)
plt.yticks([-160, -150, -140, -130, -120, -110, -100, -90, -80, -70], fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Freq. (Hz)', fontsize=12)
plt.ylabel(label1 + ' [dBc/Hz]', fontsize=12)
# plt.tight_layout()
# plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\phase_noise_calc\img\phase_noise.png', bbox_inches='tight', format='png')
plt.show()
##################################################################################

###############  CALCULATE THE  PHASE JITTER ####################################
# 1. Definir a banda de integração
f_min = 1e3   # 1 kHz
f_max = 100e3 # 100 kHz

# 2. Encontrar os índices do vetor de frequência (de forma robusta)
idx_min = np.argmin(np.abs(f - f_min))
idx_max = np.argmin(np.abs(f - f_max))

# 3. Pegar a fatia de frequência e PSD que nos interessa
f_band = f[idx_min:idx_max]
Xdb_o_band = Xdb_o[idx_min:idx_max]

# 4. Converter de dBc/Hz para rad^2/Hz
# L(f) [linear] = 10^(L(f) [dBc/Hz] / 10)
# S_phi(f) [rad^2/Hz] = 2 * L(f) [linear] (para SSB)
# Sphi_linear_band = 2 * (10**(Xdb_o_band / 10)) # DSB
Sphi_linear_band =  (10**(Xdb_o_band / 10)) # SSB

# 5. Integrar S_phi(f) sobre a banda para obter a potência de fase (rad^2)
# Usamos a regra do trapézio (np.trapz) para integrar numericamente
phase_variance_rad_sq = np.trapz(Sphi_linear_band, f_band)

# 6. Tirar a raiz quadrada para obter o RMS Phase Jitter em Radianos
phase_jitter_rad = np.sqrt(phase_variance_rad_sq)

# 7. Converter para picossegundos
phase_jitter_seconds = phase_jitter_rad / (2 * np.pi * f_c) # Use f_c (portadora real)

print(f"{Colors.BLUE}--- Jitter Integrado (1kHz a 100kHz) ---")
print(f"Jitter de Fase (RMS): {phase_jitter_rad * 1000:.4f} mrad")
print(f"Jitter de Tempo (RMS): {phase_jitter_seconds * 1e12:.4f} ps")

print(f"{Colors.RESET}")
# plt.show()