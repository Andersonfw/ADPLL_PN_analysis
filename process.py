import ctypes
from pathlib import Path as path
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



def process(t_edges, window_time, time_cut_plot_start, time_cut_plot_stop, f_required, time_cut_PN_start, fsm_file, bank_files, phe, otw, i_stop_ckv, i_stop_banks,plot_all=true, IEEE_en=false, PN_analysis=true):

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


    if PN_analysis:
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
        jitter_rms_segundos_all = np.std(jitter)

        ##############  CALCULATE THE JITTER RMS  ####################################
        print(f"{ut.Colors.BLUE}\r\n--------------------------------------------------------------------",
                "\r\nPERFORMANCE METRICS",
                "\r\n--------------------------------------------------------------------")
        

        print(f"{ut.Colors.GREEN}PN out-of-band: {ut.Colors.WHITE} {marker_dB_avg:.2f} dBc/Hz")
        print(f"{ut.Colors.GREEN}PN In-band: {ut.Colors.WHITE} {marker_dB_tdc:.2f} dBc/Hz")
        print(f"{ut.Colors.GREEN}Jitter RMS (10 kHz - 100 kHz): {ut.Colors.WHITE} {jitter_rms_segundos * 1e15:.2f} fs")
        print(f"{ut.Colors.GREEN}Jitter RMS (all BW): {ut.Colors.WHITE} {jitter_rms_segundos_all * 1e15:.2f} fs")
        print(f"{ut.Colors.GREEN}Drift Max BLE: {ut.Colors.WHITE} {ble_complice['max_drift']:.2f} kHz")
        print(f"{ut.Colors.GREEN}Drift Max HIST: {ut.Colors.WHITE} {np.max((result["f_smooth"]- f_req_array)/1e3):.2f} kHz")
        print(f"{ut.Colors.GREEN}Frequency Deviation: {ut.Colors.WHITE} {ble_complice['f_ref_deviation_hz']:.2f} Hz")
        print(f"{ut.Colors.GREEN}Peak PSD PN: {ut.Colors.WHITE} {np.max(Xdb_o):.2f} dBc/Hz")




        print(f"{ut.Colors.RESET}\r\n--------------------------------------------------------------------")


    if IEEE_en:

        plt.style.use(['science','ieee'])
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.edgecolor'] = 'lightgray'
        plt.rcParams['legend.facecolor'] = 'lightgray'


    if plot_all:

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


        if IEEE_en:

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


    return {
        "PN_out (dBc/Hz)": round(marker_dB_avg, 3),
        "PN_IN (dBc/Hz)": round(marker_dB_tdc, 3),
        "jitter_in_band (fs)": round(jitter_rms_segundos * 1e15, 3),
        "jitter_all_band (fs)": round(jitter_rms_segundos_all * 1e15, 3),
        "drift_max_ble (Hz)": round(ble_complice['max_drift'], 3),
        "drift_max_hist (Hz)": np.round(np.max((result["f_smooth"] - f_req_array)), 3),
        "f_deviation (Hz)": round(ble_complice['f_ref_deviation_hz'], 3),
        "peak_psd (dBc/Hz)": np.round(np.max(Xdb_o), 3),
        "status_drift": ble_complice['status_drift'],
        "status_rate": ble_complice['status_rate'],
        "status_f_ref_deviation": ble_complice['status_f_ref_deviation']
    }

            
