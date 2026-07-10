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


# ++++++++++++++ DEFINITIONS ++++++++++++++
IEEE_PICTURES = false
PN_ANALYSIS = TRUE

SIM_MODE = 7
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


f_required =  2.48e9 #2.39205e9 #2.402e9 #   #np.mean(fout[1])  # Hz 2.402e9
window_time = 0.5e-6  # Tamanho da janela para suavização (1us para BLE)
time_cut_PN_start = 1.8e-04
time_cut_plot_start = time_cut_PN_start#5.0e-4 #1.2e-04
time_cut_plot_stop = 250e-06

# freq = "2402"
# freq = "2440"
freq = "2480"
# freq = "2418123"
path_string = "data/"+freq
data_path = path("path_string")


# --- load Files ---
fsm_path = ut.get_latest_file("data", "fsm_states", "csv")
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
    data_path = path("data/"+freq+"/ALL_OFF")
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 5:
    # bank_path    = ut.get_latest_file(data_path, "bank_cap_SDM_off", "csv")
    # t_edges_path = ut.get_latest_file(data_path, "close_loop_edge_times_typ_SDM_off", "txt")
    t_edges_name = "close_loop_edge_times_typ"
    data_path = path("data/"+freq+"/FREF")
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 6:
    # bank_path =    ut.get_latest_file(data_path, "bank_cap_SDM_1_en", "csv")
    # t_edges_path = ut.get_latest_file(data_path, "close_loop_edge_times_typ_SDM_1_en", "txt")
    t_edges_name = "close_loop_edge_times_typ"
    data_path = path("data/"+freq+"/SDM1")
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 7:
    # bank_path =   ut.get_latest_file(data_path, "bank_cap_SDM_2_en", "csv")
    # t_edges_path = ut.get_latest_file(data_path, "close_loop_edge_times_typ_SDM_2_en", "txt")
    t_edges_name = "close_loop_edge_times_typ"
    data_path = path("data/"+freq+"/SDM2")
#-----------------------------------------------------------------------------------------------#


t_edges_path  = ut.get_latest_file(data_path, t_edges_name, "txt")
bank_path     = ut.get_latest_file(data_path, "bank_cap", "csv")
phe_path      = ut.get_latest_file(data_path, "phe", "csv")
otw_path      = ut.get_latest_file(data_path, "otw", "csv")
active_settings_path = ut.get_latest_file(data_path, "sim_historic", "csv")

bank_files = pd.read_csv(bank_path, sep=';', header=None)
t_edges = np.loadtxt(t_edges_path)
phe = pd.read_csv(phe_path, sep=';', header=None)
otw = pd.read_csv(otw_path, sep=';', header=None)
active_settings = pd.read_csv(active_settings_path, sep=';', header=None)

i_start_banks = 0 #(np.abs(bank_files[0] - time_cut_plot_start)).argmin()
i_stop_banks = (np.abs(bank_files[0].values - time_cut_plot_stop)).argmin()
i_stop_ckv = (np.abs(t_edges - time_cut_plot_stop)).argmin()
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
print(f"{ut.Colors.YELLOW}\r\n ACTIVE SETTINGS: {active_settings[1][len(active_settings)-1]} From Date {active_settings[0][len(active_settings)-1]} ")
print(f"{ut.Colors.BLUE}\r\n--------------------------------------------------------------------",)


print(f"{ut.Colors.BLUE}\r\n--------------------------------------------------------------------",
        "\r\nCALCULATE THE FREQUENCY DRIFT OF DCO OUTPUT",
        "\r\n--------------------------------------------------------------------")

result = fn.edges_convert_to_freq_out_analyses(t_edges, window_time, time_cut_plot_start, time_cut_plot_stop) 

ble_complice = fn.edges_convert_to_freq_ble_compliance(t_edges, time_cut_plot_start, f_required)


time_x_axis = result["t_edges_cut"][:len(result["f_smooth"])] * 1e6
f_req_array = np.full_like(result["f_smooth"], f_required)


if (max(fsm_file[0])) > 5:
    text_labels = ['IDLE', 'START', 'STAGE1', 'STAGE2', 'STAGE3', 'SETTLE', 'KTDC_CALIB', 'KDCO_KALIB']
else:
    text_labels = ['IDLE', 'START', 'STAGE1', 'STAGE2', 'STAGE3', 'SETTLE']

unique_y_values = sorted(fsm_file[0].unique())

print(f"{ut.Colors.BLUE}\r\n--------------------------------------------------------------------",
        "\r\nFREQUENCY DRIFT DONE",
        "\r\n--------------------------------------------------------------------")


if PN_ANALYSIS:
    ##############  CALCULATE THE PHASE NOISE OF DCO OUTPUT ####################################
    print(f"{ut.Colors.BLUE}\r\n--------------------------------------------------------------------",
            "\r\nCALCULATE THE PHASE NOISE OF DCO OUTPUT",
            "\r\n--------------------------------------------------------------------")

    

    pn_result = fn.edges_convert_to_freq_out_analyses(t_edges, window_time, time_cut_PN_start, silent=True)  # Parâmetros: tempos de borda, e tempo da janela de suavização

    t_edges_pn = pn_result["t_edges_cut"]

    T_c = 1 /pn_result["f_c"]

    t_ideal = t_edges_pn[0] + np.arange(len(t_edges_pn)) * T_c

    # Calcular o Jitter (erro de tempo acumulado)
    jitter = t_edges_pn - t_ideal
    # É vital remover a média (DC) para não distorcer o Jitter RMS depois
    jitter = jitter - np.mean(jitter)
    # jitter = 1/pn_result["f_smooth"] - t_ideal

    # nperseg: Define a Resolução em Frequência (RBW).
    # RBW ≈ fs / nperseg. Vamos mirar em 10 kHz RBW.
    # rbw_desejado = 10e3  # 10 kHz
    # nperseg = int(f_required / rbw_desejado)
    # # Limitar o tamanho da janela para não ser maior que os dados
    # if nperseg > len(jitter):
    #     nperseg = len(jitter)
    # print(f"Usando nperseg = {nperseg} para a análise PSD.")

    phase = jitter * 2 * np.pi * pn_result["f_c"]   

    Xdb_o , f = fn.fun_calc_psd(phase , pn_result["f_c"], 10e3 , 1000)  #80
    L_pn, L_freq, spurs_dbc, spurs_f, offset_spurs = fn.adpll_spectrum(phase , pn_result["f_c"])  # FFT para purius
    marker = 1e6  # Substitua pelo valor específico de frequência desejado
    indice = (np.abs(f - marker)).argmin()
    marker_dB = Xdb_o[indice]
    win_avg = 10  # Número de pontos para média móvel
    i_min = indice - win_avg
    i_max = indice + win_avg
    janela_db = Xdb_o[i_min:i_max]
    marker_dB_avg = np.mean(janela_db)
    

    marker_tdc = 10e3  # Substitua pelo valor específico de frequência desejado
    indice_tdc = (np.abs(f - marker_tdc)).argmin()
    marker_dB_tdc = Xdb_o[indice_tdc]

    print(f"{ut.Colors.BLUE}\r\n--------------------------------------------------------------------",
        "\r\nPHASE NOISE DONE",
        "\r\n--------------------------------------------------------------------")


    ##############  CALCULATE THE JITTER RMS  ####################################
    print(f"{ut.Colors.BLUE}\r\n--------------------------------------------------------------------",
            "\r\nCALCULATE THE JITTER RMS DCO OUTPUT",
            "\r\n--------------------------------------------------------------------")

    # Valor em segundos (multiplique por 1e12 para ter em ps)
    # 1. Definir os limites de integração
    f_min = 10e3   # 10 kHz
    f_max = 100e3   # 10 MHz

    # 2. Encontrar os índices do vetor 'f' que estão dentro desta banda
    indices_banda = np.where((f >= f_min) & (f <= f_max))[0]

    f_banda = f[indices_banda]
    Xdb_banda = Xdb_o[indices_banda]

    # 3. Converter de decibéis (dBc/Hz) para escala linear
    L_f_linear = 10 ** (Xdb_banda / 10.0)

    # 4. Integração numérica usando a regra do trapézio (área sob a curva)
    integral_ruido = scipy.integrate.trapezoid(L_f_linear, f_banda)

    # 5. Calcular o Jitter RMS
    # Fórmula da indústria: J_rms = (1 / (2*pi*f_c)) * sqrt(2 * integral)
    f_c = pn_result["f_c"]
    jitter_rms_segundos = (1.0 / (2 * np.pi * f_c)) * np.sqrt(2 * integral_ruido)  #multiplica por dois para considerar a banda dupla (positivo e negativo)
    jitter_rms_segundos = np.std(jitter)

    ##############  CALCULATE THE JITTER RMS  ####################################
    print(f"{ut.Colors.BLUE}\r\n--------------------------------------------------------------------",
            "\r\nPERFORMANCE METRICS",
            "\r\n--------------------------------------------------------------------")
    

    print(f"{ut.Colors.GREEN}PN out-of-band: {ut.Colors.WHITE} {marker_dB_avg:.2f} dBc/Hz")
    print(f"{ut.Colors.GREEN}PN In-band: {ut.Colors.WHITE} {marker_dB_tdc:.2f} dBc/Hz")
    print(f"{ut.Colors.GREEN}Jitter RMS (10 kHz - 100 kHz): {ut.Colors.WHITE} {jitter_rms_segundos * 1e15:.2f} fs")
    print(f"{ut.Colors.GREEN}Jitter RMS (all BW): {ut.Colors.WHITE} {jitter_rms_segundos * 1e15:.2f} fs")
    print(f"{ut.Colors.GREEN}Drift Max BLE: {ut.Colors.WHITE} {ble_complice['max_drift'] * 1e15:.2f} kHz")
    print(f"{ut.Colors.GREEN}Drift Max HIST: {ut.Colors.WHITE} {np.max((result["f_smooth"]- f_req_array)/1e3):.2f} kHz")
    print(f"{ut.Colors.GREEN}Frequency Deviation: {ut.Colors.WHITE} {ble_complice['f_ref_deviation_hz']:.2f} Hz")



    print(f"{ut.Colors.RESET}\r\n--------------------------------------------------------------------")


if IEEE_PICTURES:

    plt.style.use(['science','ieee'])
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = 'lightgray'
    plt.rcParams['legend.facecolor'] = 'lightgray'



# Tamanho padrão para 1 coluna do template IEEE (aprox. 3.5 polegadas de largura) (3.5,2.8)
# A4 size 210 × 297 mm (8.27 × 11.69 inches)
# two figures same page 2 * (6,3.8)   proporção 1.58 1.5 ~ 1.6 
pn_plot, ax_pn = plt.subplots(figsize=(5,2.8))
fsm_plot, ax_fsm = plt.subplots(figsize=(5,2.8))
spurs_plot, ax_spurs = plt.subplots(figsize=(5,2.8))
freq_plot, axs_freq = plt.subplots(nrows=2, ncols=2, figsize=(6.2, 5))
zpr_plot, axes_zpr = plt.subplots(nrows=3, ncols=1, figsize=(6.2, 5),sharex=True)
hist_plot, ax_hist = plt.subplots(figsize=(5,2.8))

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------FREQUENCY ANALYSIS------------------------------------------------------------------------------------------------------------------

FONT_LABEL = 9
FONT_TICKS = 8

label1 = "Frequency error at the output"
axs_freq[0, 0].plot(time_x_axis, (result["f_smooth"]- f_req_array)/1e3, label=label1, color='red')
axs_freq[0, 0].grid(visible=True)
axs_freq[0, 0].legend(facecolor='white', framealpha=1, fontsize=FONT_LABEL)
axs_freq[0, 0].set_xlabel(r'Time ($\mu$s)', fontsize=FONT_LABEL)
axs_freq[0, 0].set_ylabel('Frequency (kHz )', fontsize=FONT_LABEL)
axs_freq[0, 0].tick_params(axis='both', labelsize=FONT_TICKS)


label1 = "Frequency output"
axs_freq[1, 1].plot(t_edges[:i_stop_ckv] * 1e6, (result["f_smooth_full_analysis"][:i_stop_ckv] )/1e9, label=label1, color='red')
axs_freq[1, 1].grid(visible=True)
axs_freq[1, 1].legend(facecolor='white', framealpha=1, fontsize=FONT_LABEL)
axs_freq[1, 1].set_xlabel(r'Time ($\mu$s)', fontsize=FONT_LABEL)
axs_freq[1, 1].set_ylabel('Frequency (GHz)', fontsize=FONT_LABEL)
axs_freq[1, 1].tick_params(axis='both', labelsize=FONT_TICKS)

axs_freq[0, 1].set_yticks(unique_y_values, text_labels)
axs_freq[0, 1].plot( fsm_file[1] * 1e6, fsm_file[0], label=f"FSM States", color='red')
axs_freq[0, 1].grid(visible=True)
axs_freq[0, 1].legend(facecolor='white', framealpha=1, fontsize=FONT_LABEL)
axs_freq[0, 1].set_xlabel(r'Time ($\mu$s)', fontsize=FONT_LABEL)
axs_freq[0, 1].set_ylabel('State', fontsize=FONT_LABEL)
axs_freq[0, 1].tick_params(axis='x', labelsize=FONT_TICKS)
axs_freq[0, 1].tick_params(axis='y', labelsize=FONT_TICKS-2)

axs_freq[1, 0].plot(  bank_files[0] * 1e6, bank_files[1], label=f"PVT Bank", color='blue')
axs_freq[1, 0].plot(  bank_files[0] * 1e6, bank_files[2], label=f"AQ Bank", color='green')
axs_freq[1, 0].plot(  bank_files[0] * 1e6, bank_files[3], label=f"TB Bank", color='red')
axs_freq[1, 0].grid(visible=True)
axs_freq[1, 0].legend(facecolor='white', framealpha=1, fontsize=FONT_LABEL)
axs_freq[1, 0].set_xlabel(r'Time ($\mu$s)', fontsize=FONT_LABEL)
axs_freq[1, 0].set_ylabel('Capacitor bank value', fontsize=FONT_LABEL)
axs_freq[1, 0].tick_params(axis='both', labelsize=FONT_TICKS)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------FSM SEQUENCER-----------------------------------------------------------------------------------------------------------------------

FONT_LABEL = 9
FONT_TICKS = 8
ax_fsm.set_yticks(unique_y_values, text_labels)
ax_fsm.plot( fsm_file[1] * 1e6, fsm_file[0], label=f"FSM States", color='red')
ax_fsm.grid(visible=True)
ax_fsm.tick_params(axis='y', labelsize=FONT_TICKS)
ax_fsm.tick_params(axis='x', labelsize=FONT_TICKS)
# ax_fsm.legend(facecolor='white', framealpha=1, fontsize=FONT_LABEL)
ax_fsm.set_xlabel(r'Time ($\mu$s)', fontsize=FONT_LABEL)
ax_fsm.set_ylabel('State', fontsize=FONT_LABEL)

freq_plot.set_layout_engine('tight')
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------FREQ. DRIFT HISTOGRAM---------------------------------------------------------------------------------------------------------------

FONT_LABEL = 9
FONT_TICKS = 8

ax_hist.hist((result["f_smooth"]- f_req_array)/1e3, bins=40, color='tab:blue', edgecolor='black', alpha=0.8)
# 4. Customização do gráfico
ax_hist.grid(visible=True, linestyle='--', alpha=0.5)
# ax_hist.legend(fontsize=12)

# ax_hist.set_title('Exemplo de Histograma', fontsize=12, fontweight='bold')
ax_hist.set_xlabel('Frequency Drift (kHz )', fontsize=FONT_LABEL)
ax_hist.set_ylabel('Occurrences ', fontsize=FONT_LABEL)
ax_hist.grid(visible=True, linestyle=':', alpha=0.6)
ax_hist.tick_params(axis='both', labelsize=FONT_TICKS)

hist_plot.set_layout_engine('tight')
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------PHASE ERROR AND ZPR PLOTS---------------------------------------------------------------------------------------------------------

FONT_LABEL = 9
FONT_TICKS = 8
text_labels_FSM = ['OFF', 'START', r'$\alpha$ 1', r'$\alpha$ 2', r'+ $\rho$', '+ IIR']


# axes_zpr[0].set_xlabel(r'Time ($\mu$s)', fontsize=12)
# axes_zpr[0].set_ylabel('State', fontsize=12)
linha1, = axes_zpr[0].plot( fsm_file[1] * 1e6, fsm_file[0], label=f"LPF Conf. (Left Axis)", color='red')
axes_zpr[0].set_yticks(unique_y_values, text_labels_FSM)
axes_zpr[0].tick_params(axis='y', labelsize=FONT_TICKS)

ax2 = axes_zpr[0].twinx()
# ax2.set_ylabel('UNIT', fontsize=12) 
axes_zpr[0].tick_params(axis='y', labelsize=FONT_TICKS)
ax2.tick_params(axis='y', labelsize=FONT_TICKS)
linha2, = ax2.plot(phe[0][:i_stop_banks]* 1e6, phe[1][:i_stop_banks] , label=f"PHE (Right Axis)", color='blue')
linha3, = ax2.plot(phe[0][:i_stop_banks]* 1e6, otw[1][:i_stop_banks] , label=f"OTW (Right Axis)", color='green')
ax2.grid(visible=True)

linhas = [linha1, linha2, linha3]
legendas = [l.get_label() for l in linhas]
ax2.legend(linhas, legendas, facecolor='white', framealpha=1, fontsize=FONT_LABEL, loc='lower right') #(linhas, legendas, loc='upper left')

label1 = "Frequency output"
axes_zpr[1].plot(t_edges[:i_stop_ckv] * 1e6, (result["f_smooth_full_analysis"][:i_stop_ckv] )/1e9, label=label1, color='red')
axes_zpr[1].grid(visible=True)
axes_zpr[1].legend(facecolor='white', framealpha=1, fontsize=FONT_LABEL)
# axes_zpr[1].set_xlabel(r'Time ($\mu$s)', fontsize=FONT_LABEL)
axes_zpr[1].set_ylabel('Frequency (GHz)', fontsize=FONT_LABEL)
axes_zpr[1].tick_params(axis='y', labelsize=FONT_TICKS)

axes_zpr[2].plot(  bank_files[0] * 1e6, bank_files[1], label=f"PVT Bank", color='blue')
axes_zpr[2].plot(  bank_files[0] * 1e6, bank_files[2], label=f"AQ Bank", color='green')
axes_zpr[2].plot(  bank_files[0] * 1e6, bank_files[3], label=f"TB Bank", color='red')
axes_zpr[2].grid(visible=True)
axes_zpr[2].legend(facecolor='white', framealpha=1, fontsize=FONT_LABEL)
axes_zpr[2].set_xlabel(r'Time ($\mu$s)', fontsize=FONT_LABEL)
axes_zpr[2].set_ylabel('Capacitor bank value', fontsize=FONT_LABEL)
axes_zpr[2].tick_params(axis='x', labelsize=FONT_TICKS)
axes_zpr[2].tick_params(axis='y', labelsize=FONT_TICKS)
                                                    
zpr_plot.subplots_adjust(hspace=0.08)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------PHASE NOISE AND SPURS PLOTS---------------------------------------------------------------------------------------------------------

FONT_LABEL = 9
FONT_TICKS = 8

label1 = "Phase Noise"
ax_pn.semilogx(f , Xdb_o , label=label1)
ax_pn.scatter(marker, marker_dB, color='black', marker='o',s=30, zorder=3, label=f'{marker_dB_avg:.2f} dBc/Hz  @1 MHz')
ax_pn.scatter(marker_tdc, marker_dB_tdc, color='red', marker='o',s=30, zorder=3, label=f'{marker_dB_tdc:.2f} dBc/Hz  @10 kHz')
ax_pn.grid(visible=True)
ax_pn.legend( facecolor='white',framealpha=1, fontsize=FONT_LABEL)
ax_pn.set_yticks([-160, -150, -140, -130, -120, -110, -100, -90, -80, -70])
ax_pn.tick_params(axis='x', labelsize=FONT_TICKS)
ax_pn.tick_params(axis='y', labelsize=FONT_TICKS)
ax_pn.set_xlabel('Frequency Offset (Hz)', fontsize=FONT_LABEL)
ax_pn.set_ylabel(label1 + ' [dBc/Hz]', fontsize=FONT_LABEL)


label1 = "Spur Power "
ax_spurs.semilogx(spurs_f , spurs_dbc - offset_spurs , label=label1)
ax_spurs.grid(visible=True)
ax_spurs.legend( facecolor='white', framealpha=1, fontsize=FONT_LABEL)
ax_spurs.set_yticks([-170,-160, -150, -140, -130, -120, -110, -100, -90, -80, -70])
ax_spurs.set_ylim([-180, -70])
ax_spurs.tick_params(axis='x', labelsize=FONT_TICKS)
ax_spurs.tick_params(axis='y', labelsize=FONT_TICKS)
ax_spurs.set_xlabel('Frequency Offset (Hz)', fontsize=FONT_LABEL)
ax_spurs.set_ylabel(label1 + ' [dBc]', fontsize=FONT_LABEL)

pn_plot.set_layout_engine('tight')
# plt.tight_layout()  

def mostrar_alerta_windows(mensagem, titulo):
    # 0x00000000 = Botão OK apenas | 0x00000030 = Ícone de Alerta (Triângulo Amarelo)
    return ctypes.windll.user32.MessageBoxW(0, mensagem, titulo, 0x00000000 | 0x00000030)


if IEEE_PICTURES:

    while True:
        try:
            # Tenta salvar todas as figuras em lote
            pn_plot.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\pn.pdf',  dpi=600, bbox_inches='tight', format='pdf')
            spurs_plot.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\spurs.pdf',  dpi=600, bbox_inches='tight', format='pdf')
            freq_plot.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\freq.pdf',  dpi=600, bbox_inches='tight', format='pdf')
            zpr_plot.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\zpr.pdf',  dpi=600, bbox_inches='tight', format='pdf')
            hist_plot.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\freq_drift_hist.pdf',  dpi=600, bbox_inches='tight', format='pdf')
            fsm_plot.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\FSM.pdf',  dpi=600, bbox_inches='tight', format='pdf')
            
            # Se chegou aqui sem dar erro, avisa no terminal e sai do loop
            print(f"{ut.Colors.GREEN}Sucesso! Todos os gráficos em PDF foram salvos.")
            break
            
        except (PermissionError, PermissionError) as e:
            # Captura o erro de arquivo travado/aberto (geralmente PermissionError ou OSError no Windows)
            mensagem = "Erro ao salvar os gráficos!\n\nUm ou mais arquivos PDF estão abertos no leitor de PDF.\nPor favor, feche todos eles e clique em OK para tentar novamente."
            titulo = "Arquivo PDF Bloqueado"

            print(f"{ut.Colors.RED}Erro ao salvar os gráficos!")
            print(f"{ut.Colors.YELLOW}Detalhes do erro: {e}")
            # Pausa o código e exibe o pop-up na tela
            mostrar_alerta_windows(mensagem, titulo)                                                                                
        
    print(f"{ut.Colors.RESET}")



else:
    plt.show()
        
        
    # plt.figure(figsize=(6,4), dpi=600)
    # label1 = "Frequency error at the output"
    # plt.plot(time_x_axis, (result["f_smooth"] - f_req_array)/1e3, label=label1, color='black')
    # plt.grid(visible=True)  
    # plt.legend(facecolor='white', framealpha=1, fontsize=12)
    # plt.xticks(fontsize=12)  
    # plt.yticks(fontsize=12)  
    # plt.xlabel(r'Time ($\mu$s)', fontsize=12)
    # plt.ylabel('Frequency (kHz)', fontsize=12)  
    # # plt.tight_layout()  
    # plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\freq_drift.png', bbox_inches='tight', format='png')

    # plt.figure(figsize=(6,4), dpi=600)
    # label1 = "Frequency output"
    # plt.plot(t_edges[:i_stop_ckv] * 1e6, (result["f_smooth_full_analysis"][:i_stop_ckv] )/1e3, label=label1, color='red')
    # plt.grid(visible=True)
    # plt.legend(facecolor='white', framealpha=1, fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.xlabel(r'Time ($\mu$s)', fontsize=12)
    # plt.ylabel('Frequency (kHz)', fontsize=12)
    # # plt.tight_layout()
    # plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\freq_out.png', bbox_inches='tight', format='png')


    # plt.figure(figsize=(6,4), dpi=600)
    # label1 = "FSM States"
    # plt.plot(fsm_file[1] * 1e6, fsm_file[0], label=label1, color='black')
    # plt.grid(visible=True)
    # plt.legend(facecolor='white', framealpha=1, fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.xlabel(r'Time ($\mu$s)', fontsize=12)
    # plt.ylabel('State', fontsize=12)
    # # plt.tight_layout()
    # plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\FSM.png', bbox_inches='tight', format='png')

    # plt.figure(figsize=(6,4), dpi=600)
    # plt.plot(  bank_files[0] * 1e6, bank_files[1], label=f"PVT Bank", color='blue')
    # plt.plot(  bank_files[0] * 1e6, bank_files[2], label=f"AQ Bank", color='green')
    # plt.plot(  bank_files[0] * 1e6, bank_files[3], label=f"TB Bank", color='red')
    # plt.grid(visible=True)
    # plt.legend(facecolor='white', framealpha=1, fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.xlabel(r'Time ($\mu$s)', fontsize=12)
    # plt.ylabel('Capacitor bank value', fontsize=12)
    # # plt.tight_layout()
    # plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\dco_bank.png', bbox_inches='tight', format='png')


    # plt.figure(figsize=(6,4), dpi=600)
    # label1 = "Phase Noise"
    # plt.semilogx(f , Xdb_o , label=label1)
    # plt.scatter(marker, marker_dB, color='blue', marker='o', label=f'{marker_dB:.2f} dBc/Hz  @1 MHz')
    # # plt.scatter(marker_tdc, marker_dB_tdc, color='red', marker='o', label=f'{marker_dB_tdc:.2f} dBc/Hz  @10 kHz')
    # plt.grid(visible=True)
    # plt.legend(facecolor='white', framealpha=1, fontsize=12)
    # plt.yticks([-160, -150, -140, -130, -120, -110, -100, -90, -80, -70], fontsize=12)
    # plt.xlabel('Freq. (Hz)', fontsize=12)
    # plt.ylabel(label1 + ' [dBc/Hz]', fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\pn.png', bbox_inches='tight', format='png')


    
