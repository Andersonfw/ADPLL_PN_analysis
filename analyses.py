from pathlib import Path as path
from pickle import TRUE

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import locale
import datetime
import os
import scienceplots as sp
import pandas as pd
from sympy import false, true

import function as fn

class Colors:
    WHITE = '\033[97m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    YELLOW = '\033[0;33m'
    BLUE = "\033[0;34m"

'''------------------------------------------------------------------------------------------------------- '''

# ++++++++++++++ DEFINITIONS ++++++++++++++
IEEE_PICTURES = false
PN_ANALYSIS = TRUE

SIM_MODE = 1
""" 
    SIM_MODE = 0: Typical Corner
    SIM_MODE = 1: Slow Corner
    SIM_MODE = 2: Fast Corner
    SIM_MODE = 3: FUTURE USE
    SIM_MODE = 4: FUTURE USE

    SIM_MODE = 5: Sem SDM
    SIM_MODE = 6: SDM 1 Ordem
    SIM_MODE = 7: SDM 2 Ordem 
    SIM_MODE = x: DEFAULT SIMULATION
"""


f_required =  2.440e9 #2.39205e9 #2.402e9 #   #np.mean(fout[1])  # Hz 2.402e9
window_time = 0.5e-6  # Tamanho da janela para suavização (1us para BLE)
time_cut_PN = 1.4e-04
time_cut_f_drift = time_cut_PN#5.0e-4 #1.2e-04



# --- load Files ---
fsm_file = pd.read_csv("data/fsm_states.csv", sep=';', header=None)

t_edges_path = path("data/close_loop_edge_times_sim.txt")
bank_path = path("data/bank_cap.csv")
phe_path = path("data/phe.csv")
ntw_path = path("data/ntw.csv")

if  SIM_MODE == 0:
    t_edges_path = path("data/close_loop_edge_times_typ.txt")  
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 1:
    t_edges_path =path("data/close_loop_edge_times_worst.txt")
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 2:
    t_edges_path = path("data/close_loop_edge_times_best.txt")
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 5:
    bank_path    = path("data/bank_cap_SDM_off.csv")
    t_edges_path = path("data/close_loop_edge_times_typ_SDM_off.txt")
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 6:
    bank_path =    path("data/bank_cap_SDM_1_en.csv")
    t_edges_path = path("data/close_loop_edge_times_typ_SDM_1_en.txt")
#-----------------------------------------------------------------------------------------------#
elif SIM_MODE == 7:
    bank_path =   path("data/bank_cap_SDM_2_en.csv")
    t_edges_path = path("data/close_loop_edge_times_typ_SDM_2_en.txt")



bank_files = pd.read_csv(bank_path, sep=';', header=None)
t_edges = np.loadtxt(t_edges_path)
phe = pd.read_csv(phe_path, sep=';', header=None)
ntw = pd.read_csv(ntw_path, sep=';', header=None)

fsm_file.loc[len(fsm_file)] = [5,  bank_files[0].iloc[-1]]  # Adiciona um ponto extra para manter a linha até o final do tempo

print(f"{Colors.BLUE}\r\n--------------------------------------------------------------------"
        "\r\nFILES LOADED",)
print(f"{Colors.YELLOW}\r\n Egdes_file: {t_edges_path.name}")
print(f"{Colors.YELLOW}\r\n Bank_file: {bank_path.name}")
print(f"{Colors.BLUE}\r\n--------------------------------------------------------------------",)


print(f"{Colors.BLUE}\r\n--------------------------------------------------------------------",
        "\r\nCALCULATE THE FREQUENCY DRIFT OF DCO OUTPUT",
        "\r\n--------------------------------------------------------------------")

result = fn.edges_convert_to_freq_out_analyses(t_edges, window_time, time_cut_f_drift) 

result_t2 = fn.edges_convert_to_freq_ble_compliance(t_edges, time_cut_f_drift, f_required)

indice_edges_PN = (np.abs(t_edges - time_cut_PN)).argmin()
t_edges_pn = t_edges[indice_edges_PN:]  

indice_edges_f_drift = (np.abs(t_edges - time_cut_PN)).argmin()

time_x_axis = result["t_edges_cut"][:len(result["f_smooth"])] * 1e6
f_req_array = np.full_like(result["f_smooth"], f_required)


if (max(fsm_file[0])) > 5:
    text_labels = ['IDLE', 'START', 'STAGE1', 'STAGE2', 'STAGE3', 'SETTLE', 'KTDC_CALIB', 'KDCO_KALIB']
else:
    text_labels = ['IDLE', 'START', 'STAGE1', 'STAGE2', 'STAGE3', 'SETTLE']

unique_y_values = sorted(fsm_file[0].unique())

print(f"{Colors.BLUE}\r\n--------------------------------------------------------------------",
        "\r\nFREQUENCY DRIFT DONE",
        "\r\n--------------------------------------------------------------------")


if PN_ANALYSIS:
    ##############  CALCULATE THE PHASE NOISE OF DCO OUTPUT ####################################
    print(f"{Colors.BLUE}\r\n--------------------------------------------------------------------",
            "\r\nCALCULATE THE PHASE NOISE OF DCO OUTPUT",
            "\r\n--------------------------------------------------------------------")

    

    pn_result = fn.edges_convert_to_freq_out_analyses(t_edges_pn, window_time, time_cut_PN, silent=True)  # Parâmetros: tempos de borda, e tempo da janela de suavização

    T_c = 1 /pn_result["f_c"]

    t_ideal = t_edges_pn[0] + np.arange(len(t_edges_pn)) * T_c
    # t_ideal = T_c + np.arange(len(1/pn_result["f_smooth"])) * T_c

    # Calcular o Jitter (erro de tempo acumulado)
    jitter = t_edges_pn - t_ideal
    # É vital remover a média (DC) para não distorcer o Jitter RMS depois
    jitter = jitter - np.mean(jitter)
    # jitter = 1/pn_result["f_smooth"] - t_ideal


    # nperseg: Define a Resolução em Frequência (RBW).
    # RBW ≈ fs / nperseg. Vamos mirar em 10 kHz RBW.
    rbw_desejado = 10e3  # 10 kHz
    nperseg = int(f_required / rbw_desejado)
    # Limitar o tamanho da janela para não ser maior que os dados
    if nperseg > len(jitter):
        nperseg = len(jitter)
    print(f"Usando nperseg = {nperseg} para a análise PSD.")

    phase = jitter * 2 * np.pi * pn_result["f_c"]
    b = 1
    a = np.array([1 , -1])
    x = signal.lfilter([b], a,  phase - np.mean(phase))

   

    Xdb_o , f = fn.fun_calc_psd(phase , pn_result["f_c"], 10e3 , 1000)  #80
    Xdb_o1 , f1 = fn.adpll_spectrum(phase , pn_result["f_c"])  
    marker = 1e6  # Substitua pelo valor específico de frequência desejado
    indice = (np.abs(f - marker)).argmin()
    marker_dB = Xdb_o[indice]

    marker_tdc = 10e3  # Substitua pelo valor específico de frequência desejado
    indice_tdc = (np.abs(f - marker_tdc)).argmin()
    marker_dB_tdc = Xdb_o[indice_tdc]

    print(f"{Colors.BLUE}\r\n--------------------------------------------------------------------",
        "\r\nPHASE NOISE DONE",
        "\r\n--------------------------------------------------------------------")


    ##############  CALCULATE THE JITTER RMS  ####################################
    print(f"{Colors.BLUE}\r\n--------------------------------------------------------------------",
            "\r\nCALCULATE THE JITTER RMS DCO OUTPUT",
            "\r\n--------------------------------------------------------------------")

    # Valor em segundos (multiplique por 1e12 para ter em ps)
    # 1. Definir os limites de integração
    f_min = 10e3   # 10 kHz
    f_max = 20e6   # 10 MHz

    # 2. Encontrar os índices do vetor 'f' que estão dentro desta banda
    indices_banda = np.where((f >= f_min) & (f <= f_max))[0]

    f_banda = f[indices_banda]
    Xdb_banda = Xdb_o[indices_banda]

    # 3. Converter de decibéis (dBc/Hz) para escala linear
    L_f_linear = 10 ** (Xdb_banda / 10.0)

    # 4. Integração numérica usando a regra do trapézio (área sob a curva)
    integral_ruido = np.trapz(L_f_linear, f_banda)

    # 5. Calcular o Jitter RMS
    # Fórmula da indústria: J_rms = (1 / (2*pi*f_c)) * sqrt(2 * integral)
    f_c = pn_result["f_c"]
    jitter_rms_segundos = (1.0 / (2 * np.pi * f_c)) * np.sqrt(2 * integral_ruido)

    print(f"{Colors.GREEN}Jitter RMS (10 kHz - 10 MHz): {jitter_rms_segundos * 1e15:.2f} fs")
    
    jitter_rms_segundos = np.std(jitter)
    print(f"{Colors.GREEN}Jitter RMS (all BW): {jitter_rms_segundos * 1e15:.2f} fs")


    print(f"{Colors.RESET}\r\n--------------------------------------------------------------------")

if IEEE_PICTURES:

    plt.style.use(['science','ieee'])
    plt.style.use(['science','ieee'])
    plt.rcParams['legend.frameon'] = True  # Mostrar a moldura da legenda
    plt.rcParams['legend.edgecolor'] = 'lightgray'  # Cor da borda da legenda
    plt.rcParams['legend.facecolor'] = 'lightgray'  # Cor do fundo da legenda


    plt.figure(figsize=(6,4), dpi=600)
    label1 = "Frequency error at the output"
    plt.plot(time_x_axis, (result["f_smooth"] - f_req_array)/1e3, label=label1, color='black')
    plt.grid(visible=True)
    plt.legend(facecolor='white', framealpha=1, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time (us)', fontsize=12)
    plt.ylabel('Frequency (kHz)', fontsize=12)
    # plt.tight_layout()
    plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\freq_drift.png', bbox_inches='tight', format='png')

    plt.figure(figsize=(6,4), dpi=600)
    label1 = "Frequency output"
    plt.plot(t_edges[:len(result["f_smooth_full_analysis"])] * 1e6, (result["f_smooth_full_analysis"] )/1e3, label=label1, color='black')
    plt.grid(visible=True)
    plt.legend(facecolor='white', framealpha=1, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time (us)', fontsize=12)
    plt.ylabel('Frequency (kHz)', fontsize=12)
    # plt.tight_layout()
    plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\freq_out.png', bbox_inches='tight', format='png')


    plt.figure(figsize=(6,4), dpi=600)
    label1 = "FSM States"
    plt.plot(fsm_file[1] * 1e6, fsm_file[0], label=label1, color='black')
    plt.grid(visible=True)
    plt.legend(facecolor='white', framealpha=1, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time (us)', fontsize=12)
    plt.ylabel('STATE', fontsize=12)
    # plt.tight_layout()
    plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\FSM.png', bbox_inches='tight', format='png')

    plt.figure(figsize=(6,4), dpi=600)
    plt.plot(  bank_files[0] * 1e6, bank_files[1], label=f"PVT Bank", color='blue')
    plt.plot(  bank_files[0] * 1e6, bank_files[2], label=f"AQ Bank", color='green')
    plt.plot(  bank_files[0] * 1e6, bank_files[3], label=f"TB Bank", color='red')
    plt.grid(visible=True)
    plt.legend(facecolor='white', framealpha=1, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time (us)', fontsize=12)
    plt.ylabel('Capacitor bank value', fontsize=12)
    # plt.tight_layout()
    plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\dco_bank.png', bbox_inches='tight', format='png')


    plt.figure(figsize=(6,4), dpi=600)
    label1 = "Phase Noise"
    plt.semilogx(f , Xdb_o , label=label1)
    plt.scatter(marker, marker_dB, color='blue', marker='o', label=f'{marker_dB:.2f} dBc/Hz  @1 MHz')
    # plt.scatter(marker_tdc, marker_dB_tdc, color='red', marker='o', label=f'{marker_dB_tdc:.2f} dBc/Hz  @10 kHz')
    plt.grid(visible=True)
    plt.legend(facecolor='white', framealpha=1, fontsize=12)
    plt.yticks([-160, -150, -140, -130, -120, -110, -100, -90, -80, -70], fontsize=12)
    plt.xlabel('Freq. (Hz)', fontsize=12)
    plt.ylabel(label1 + ' [dBc/Hz]', fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\pn.png', bbox_inches='tight', format='png')


    

else:
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    label1 = "Frequency error at the output"
    axs[0, 0].plot(time_x_axis, (result["f_smooth"]- f_req_array)/1e3, label=label1, color='red')
    axs[0, 0].grid(visible=True)
    axs[0, 0].legend(facecolor='white', framealpha=1, fontsize=12)
    axs[0, 0].set_xlabel('Time (us)', fontsize=12)
    axs[0, 0].set_ylabel('Frequency (kHz )', fontsize=12)


    label1 = "Frequency output"
    axs[1, 1].plot(t_edges[:len(result["f_smooth_full_analysis"])] * 1e6, (result["f_smooth_full_analysis"] )/1e3, label=label1, color='red')
    axs[1, 1].grid(visible=True)
    axs[1, 1].legend(facecolor='white', framealpha=1, fontsize=12)
    axs[1, 1].set_xlabel('Time (us)', fontsize=12)
    axs[1, 1].set_ylabel('Frequency (MHz )', fontsize=12)

    axs[0, 1].set_yticks(unique_y_values, text_labels)
    axs[0, 1].plot( fsm_file[1] * 1e6, fsm_file[0], label=f"FSM States", color='red')
    axs[0, 1].grid(visible=True)
    axs[0, 1].legend(facecolor='white', framealpha=1, fontsize=12)
    axs[0, 1].set_xlabel('Time (us)', fontsize=12)
    axs[0, 1].set_ylabel('STATE', fontsize=12)
 
    axs[1, 0].plot(  bank_files[0] * 1e6, bank_files[1], label=f"PVT Bank", color='blue')
    axs[1, 0].plot(  bank_files[0] * 1e6, bank_files[2], label=f"AQ Bank", color='green')
    axs[1, 0].plot(  bank_files[0] * 1e6, bank_files[3], label=f"TB Bank", color='red')
    axs[1, 0].grid(visible=True)
    axs[1, 0].legend(facecolor='white', framealpha=1, fontsize=12)
    axs[1, 0].set_xlabel('Time (us)', fontsize=12)
    axs[1, 0].set_ylabel('Capacitor bank value', fontsize=12)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

    axes[0].set_xlabel('Time (us)', fontsize=12)
    axes[0].set_ylabel('STATE', fontsize=12)
    linha1, = axes[0].plot( fsm_file[1] * 1e6, fsm_file[0], label=f"FSM States", color='red')
    axes[0].set_yticks(unique_y_values, text_labels)
    axes[0].grid(visible=True)
    axes[0].tick_params(axis='y')

    ax2 = axes[0].twinx()

    linha2, = ax2.plot(phe[0]* 1e6, phe[1] , label=f"PHE", color='blue')
    linha3, = ax2.plot(phe[0]* 1e6, ntw[1] , label=f"NTW", color='green')
    ax2.grid(visible=True)
    ax2.tick_params(axis='y')

    plt.title('Sinais Sobrepostos com Diferentes Escalas no Eixo Y')

    linhas = [linha1, linha2, linha3]
    legendas = [l.get_label() for l in linhas]
    axes[0].legend(linhas, legendas, loc='upper left')

    label1 = "Frequency output"
    axes[1].plot(t_edges[:len(result["f_smooth_full_analysis"])] * 1e6, (result["f_smooth_full_analysis"] )/1e3, label=label1, color='red')
    axes[1].grid(visible=True)
    axes[1].legend(facecolor='white', framealpha=1, fontsize=12)
    axes[1].set_xlabel('Time (us)', fontsize=12)
    axes[1].set_ylabel('Frequency (MHz )', fontsize=12)

    axes[2].plot(  bank_files[0] * 1e6, bank_files[1], label=f"PVT Bank", color='blue')
    axes[2].plot(  bank_files[0] * 1e6, bank_files[2], label=f"AQ Bank", color='green')
    axes[2].plot(  bank_files[0] * 1e6, bank_files[3], label=f"TB Bank", color='red')
    axes[2].grid(visible=True)
    axes[2].legend(facecolor='white', framealpha=1, fontsize=12)
    axes[2].set_xlabel('Time (us)', fontsize=12)
    axes[2].set_ylabel('Capacitor bank value', fontsize=12)
                                                        

    # label1 = "PHASE ERROR at the output PHASE DETECTOR"
    # axs[0, 0].plot(phe[0]* 1e6, phe[1] / 1e6, label=label1, color='red')
    # axs[0, 0].grid(visible=True)
    # axs[0, 0].legend(facecolor='white', framealpha=1, fontsize=12)
    # axs[0, 0].set_xlabel('Time (us)', fontsize=12)
    # # axs[0, 0].set_ylabel('Frequency (kHz )', fontsize=12)

    # label1 = "LOOP FILTER OUTPUT VALUE"
    # axs[1, 1].plot(phe[0] * 1e6,ntw[0] / 1e6, label=label1, color='red')
    # axs[1, 1].grid(visible=True)
    # axs[1, 1].legend(facecolor='white', framealpha=1, fontsize=12)
    # axs[1, 1].set_xlabel('Time (us)', fontsize=12)
    # # axs[1, 1].set_ylabel('Frequency (MHz )', fontsize=12)

    # axs[0, 1].set_yticks(unique_y_values, text_labels)
    # axs[0, 1].plot( fsm_file[1] * 1e6, fsm_file[0], label=f"FSM States", color='red')
    # axs[0, 1].grid(visible=True)
    # axs[0, 1].legend(facecolor='white', framealpha=1, fontsize=12)
    # axs[0, 1].set_xlabel('Time (us)', fontsize=12)
    # axs[0, 1].set_ylabel('STATE', fontsize=12)



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    plt.figure()
    label1 = "Phase Noise"
    plt.semilogx(f , Xdb_o , label=label1)
    plt.scatter(marker, marker_dB, color='black', marker='o', label=f'{marker_dB:.2f} dBc/Hz  @1 MHz')
    plt.scatter(marker_tdc, marker_dB_tdc, color='red', marker='o', label=f'{marker_dB_tdc:.2f} dBc/Hz  @10 kHz')
    plt.grid(visible=True)
    plt.legend()
    plt.yticks([-160, -150, -140, -130, -120, -110, -100, -90, -80, -70], fontsize=8)
    plt.xlabel('Freq. (Hz)', fontsize=8)
    plt.ylabel(label1 + ' [dBc/Hz]', fontsize=8)

    plt.show()