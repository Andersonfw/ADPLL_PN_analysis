import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import locale
import datetime
import os
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

# --- 1. INGREDIENTES ---
# Parâmetros do seu sistema
f_carrier = 4.9e9  # Frequência da portadora em Hz
FN_DECIM = 200.0   # Fator de decimação
fs = f_carrier / FN_DECIM  # Frequência de amostragem dos dados
Ideal_period = 1 / f_carrier
desv_time = 0
# Carregar os dados de jitter do arquivo
try:
    period_t = np.loadtxt('data/period_dither.txt') * 1e-15 
    N = len(period_t) # Número de amostras
    freq = 1 / period_t
    print(f"Arquivo carregado com sucesso. Número de amostras: {N}")
except IOError:
    print("Erro: Arquivo 'total_jitter.txt' não encontrado.")
    print("Gerando dados de exemplo para demonstração.")
    # Gerando um ruído de exemplo caso o arquivo não exista
    N = 100000
    noise_white = np.random.randn(N) * 1e-15 # Ruído branco
    noise_flicker = np.cumsum(np.random.randn(N)) * 5e-18 # Ruído 1/f simplificado
    period_t = noise_white + noise_flicker

print(f"{Colors.BLUE}Max frequency: ",np.max(freq)/1e9,"@ ", np.argmax(freq), "Mean Freq: ",np.mean(freq)/1e9, " Min frequency: ",np.min(freq)/1e9,"@ ", np.argmin(freq), " MHz")

 ################ PHASE ERROR IN RAD/S #################
print(f"{Colors.BLUE}--------------------------------------------------------------------,"
        "\r\nCalculation of the fase erro in rad/s at 2.4 GHz",
        "\r\n--------------------------------------------------------------------")
tckv = np.array(period_t)
phase = np.zeros(len(tckv)) 
for i in range(len(tckv)):
    diff = Ideal_period - (tckv[i])
    phase[i] = diff * 2*np.pi * f_carrier

print(f"{Colors.BLUE}Max error: ",np.max(phase), "Mean error: ",np.mean(phase), " rad/s")
####################################################### 
# --- 2. MATEMÁTICA REVERSA ---

# Passo 1: Converter Jitter para Desvio de Fase
# phase = 2 * np.pi * f_carrier * period_t


b = 1
a = np.array([1 , -1])
x = signal.lfilter([b], a,  phase - np.mean(phase))
##################################################################################

###############  CALCULATE THE PHASE NOISE OF DCO OUTPUT ####################################
print(f"{Colors.BLUE}\r\n--------------------------------------------------------------------",
        "\r\nCALCULATE THE PHASE NOISE OF DCO OUTPUT",
        "\r\n--------------------------------------------------------------------")
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
phase_jitter_seconds = phase_jitter_rad / ( 2 * np.pi * f_carrier)
phase_jitter_deg =  phase_jitter_seconds / ((1/f_carrier)/360)
print(f"{Colors.BLUE}Phase jitter: ", phase_jitter_rad * 1000, " mrad/s ---- ", phase_jitter_seconds * 1e12, " ps  ----", "Jitter in degrees: ", phase_jitter_deg, "deg")

plt.show()















##############################################################################################################

# # Passo 2: Calcular a FFT (usando uma janela para melhores resultados)
# # A janela de Hann reduz artefatos nas bordas do espectro
# window = np.hanning(N)
# phi_t_windowed = phi_t * window
# Y_k = np.fft.fft(phi_t_windowed)

# # Passo 3: Calcular a PSD
# # O fator de normalização da janela é sum(window**2)
# psd_phi = (1 / (fs * np.sum(window**2))) * np.abs(Y_k)**2

# # A FFT produz um espectro espelhado, então pegamos apenas a primeira metade
# # e multiplicamos por 2 para conservar a energia total.
# n_one_side = N // 2
# psd_phi_one_side = psd_phi[:n_one_side] * 2

# # Passo 4: Criar o Eixo de Frequência de Offset
# freq_axis = np.fft.fftfreq(N, 1/fs)[:n_one_side]

# # Passo 5: Converter a PSD para dBc/Hz
# # O fator 0.5 converte para o padrão de banda lateral única (SSB)
# L_f = 10 * np.log10(0.5 * psd_phi_one_side)

# # --- 3. PLOTAR O GRÁFICO ---

# plt.figure(figsize=(10, 6))
# # Plotamos em escala log-log, que é o padrão para ruído de fase
# # Ignoramos o primeiro ponto (f=0) que é o componente DC
# plt.semilogx(freq_axis[1:], L_f[1:])
# plt.title('Gráfico de Ruído de Fase do DCO')
# plt.xlabel('Frequência de Offset [Hz]')
# plt.ylabel('Ruído de Fase [dBc/Hz]')
# plt.grid(True, which="both")
# plt.show()