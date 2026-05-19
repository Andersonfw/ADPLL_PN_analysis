import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import locale
import datetime
import os

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
t_edges = np.loadtxt("edge_times.txt")
print(f"Carregados {len(t_edges)} tempos de borda.")

# --- 3. Calcular Frequência Média e Jitter ---
N_edges = len(t_edges)
# Frequência média (portadora)
f_c = (N_edges - 1) / (t_edges[-1] - t_edges[0])
# Período médio

print(f"Frequência média da portadora (f_c): {f_c/1e9:.6f} GHz")

F_d = 4.8e9  # Frequência desejada em Hz
T_c = 1.0 / F_d

# Criar tempos de borda ideais, começando do primeiro tempo real
t_ideal = T_c + np.arange(N_edges) * T_c

# Calcular o Jitter (erro de tempo acumulado)
# Esta é a resposta para "Como calcular o jitter?"
jitter = t_edges - t_ideal

# --- 4. Calcular a PSD do Jitter (Sj) ---
# Esta é a resposta para "Como calcular a PSD?"

# Parâmetros para o Welch:
# fs: A taxa de amostragem do nosso vetor 'jitter'. Como temos uma
#     amostra de jitter por borda de clock, fs = f_c
fs = F_d

# nperseg: Define a Resolução em Frequência (RBW).
# RBW ≈ fs / nperseg. Vamos mirar em 10 kHz RBW.
rbw_desejado = 10e3  # 10 kHz
nperseg = int(fs / rbw_desejado)
# Limitar o tamanho da janela para não ser maior que os dados
if nperseg > len(jitter):
    nperseg = len(jitter)
print(f"Usando nperseg = {nperseg} para a análise PSD.")

# Calcular a PSD do Jitter (Sj). Unidades: s^2/Hz
# scaling='density' nos dá a PSD (V^2/Hz, ou no nosso caso s^2/Hz)



##############  CALCULATE THE PHASE NOISE OF DCO OUTPUT ####################################
print(f"{Colors.BLUE}\r\n--------------------------------------------------------------------",
        "\r\nCALCULATE THE PHASE NOISE OF DCO OUTPUT",
        "\r\n--------------------------------------------------------------------")

phase = jitter * 2 * np.pi * f_c
b = 1
a = np.array([1 , -1])
x = signal.lfilter([b], a,  phase - np.mean(phase))

Xdb_o , f = fn.fun_calc_psd(x , fs, 100e3 , 500)  #80
marker = 1e6  # Substitua pelo valor específico de frequência desejado
indice = np.where(f == marker)[0][0]
marker_dB = Xdb_o[indice]

marker_tdc = 1e3  # Substitua pelo valor específico de frequência desejado
indice_tdc = np.where(f == marker_tdc)[0][0]
marker_dB_tdc = Xdb_o[indice_tdc]

plt.figure()
label1 = "Phase Noise"
plt.semilogx(f , Xdb_o , label=label1)
plt.scatter(marker, marker_dB, color='black', marker='o', label=f'{marker_dB:.2f} dBc/Hz  @1 MHz')
plt.scatter(marker_tdc, marker_dB_tdc, color='red', marker='o', label=f'{marker_dB_tdc:.2f} dBc/Hz  @100 kHz')
plt.grid(visible=True)
plt.legend()
plt.yticks([-160, -150, -140, -130, -120, -110, -100, -90, -80, -70], fontsize=8)
plt.xlabel('Freq. (Hz)', fontsize=8)
plt.ylabel(label1 + ' [dBc/Hz]', fontsize=8)

##################################################################################

###############  CALCULATE THE  PHASE JITTER ####################################
phase_jitter_rad = 0

phase_jitter_seconds = 0
phase_jitter_dbc = 0
index = np.where(f == 50e3)[0][0]
Bw = 100e3 - 1e3
phase_jitter_dbc = 10 **((Xdb_o[index] + 10*np.log10(Bw))/10)
# print("phase_jitter_dbc:  ", phase_jitter_dbc, " Value in dBc: ", Xdb_o[index])
phase_jitter_rad = np.sqrt(2* phase_jitter_dbc)
phase_jitter_seconds = phase_jitter_rad / ( 2 * np.pi * f_c)
phase_jitter_deg =  phase_jitter_seconds / ((1/f_c)/360)
print(f"{Colors.BLUE}Phase jitter: ", phase_jitter_rad * 1000, " mrad/s ---- ", phase_jitter_seconds * 1e12, " ps  ----", "Jitter in degrees: ", phase_jitter_deg, "deg")

plt.show()




# f, Sj_linear = signal.welch(
#     jitter,
#     fs=fs,
#     nperseg=nperseg,
#     scaling='density'
# )

# # --- 5. Converter PSD do Jitter (Sj) para Ruído de Fase (L(f)) ---
# # S_phi(f) [rad^2/Hz] = (2*pi*f_c)^2 * S_j(f) [s^2/Hz]
# Sphi_linear = ((2 * np.pi * f_c)**2) * Sj_linear

# # L(f) [dBc/Hz] = 10 * log10( S_phi(f) / 2 )
# # (Dividimos por 2 para obter o "single-sideband" (SSB) phase noise)

# # Usamos np.maximum para evitar log(0) caso o ruído seja zero
# L_f_dB = 10 * np.log10(np.maximum(Sphi_linear / 2, 1e-25))


# # --- 6. Plotar os Resultados ---
# # Esta é a resposta para "Como plotar?"

# print("Gerando gráficos...")
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
# fig.suptitle(f"Análise de Ruído de Fase (Portadora: {f_c/1e9:.3f} GHz)", fontsize=16)

# # Plot 1: Jitter Acumulado vs. Tempo
# ax1.plot(t_edges, jitter * 1e12) # Plotar jitter em picossegundos
# ax1.set_title("Jitter Acumulado vs. Tempo")
# ax1.set_xlabel("Tempo (s)")
# ax1.set_ylabel("Jitter (ps)")
# ax1.grid(True)

# # Plot 2: Ruído de Fase (L(f)) vs. Frequência (log)
# # Ignoramos f[0] (o componente DC) para o plot semilogx
# ax2.semilogx(f[1:], L_f_dB[1:])
# ax2.set_title("PSD do Ruído de Fase (L(f))")
# ax2.set_xlabel("Frequência Offset (Hz)")
# ax2.set_ylabel("Ruído de Fase (dBc/Hz)")
# ax2.grid(True, which="both") # Grid para eixos log e linear

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig("phase_noise_analysis.png")
# print("Análise concluída. Gráfico salvo como 'phase_noise_analysis.png'")