import numpy as np
import matplotlib.pyplot as plt
from sympy import false, true

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

IEEE_PICTURES = TRUE  # Set to True to generate IEEE-style plots, False for default style
ALL_IN_ONE = true

# --- Parâmetros do ADPLL ---
fR = 26e6   # Reference Frequency
fout=4.8e9  # RF Frequency

DELTA_F = 1e6
TDC_RES = 12e-12
FRES_DCO = 31.25e3
TREF = 1/fR
FDCO = fout/2
TDCO = int(1/FDCO / TDC_RES) * TDC_RES
noise_DCO_time = (1/12) * ((FRES_DCO/DELTA_F)**2) * TREF * (np.sinc(DELTA_F/fR)**2)
noise_DCO_DB = 10*np.log10(noise_DCO_time)
noise_TDC_time = ( ( (2*np.pi)**2) / 12) * (TDC_RES/TDCO)**2 * TREF
noise_TDC_DB = 10*np.log10(noise_TDC_time)

FCW = fout/fR  


S_tdc_floor = noise_TDC_DB  # Noise floor of the TDC estimated in dBc/Hz
S_dco_floor = noise_DCO_DB  # Noise floor of the DCO estimated in dBc/Hz

#LPF CONFIS
configs = [
    {"alpha": 2**-5, "rho": 2**-10},
    {"alpha": 2**-5, "rho": 2**-11},
    {"alpha": 2**-5, "rho": 2**-12},
    {"alpha": 2**-5, "rho": 2**-13},
    {"alpha": 2**-5, "rho": 2**-14}
]

#IIR COEFICIENTS CONFIGS
irr_configs = [
    {"lambda1": 2**-2, "lambda2": 2**-2, "lambda3": 2**-2, "lambda4": 2**-2},
]

plot_color = [
    {"color_p": "red"},
    {"color_p": "blue"},
    {"color_p": "black"},
    {"color_p": "green"},
    {"color_p": "orange"}
]
# Frequency offset vector (from 10 Hz to 100 MHz in logarithmic scale)
f_offset = np.logspace(2, 8, 1000)
s = 2j * np.pi * f_offset  # s = j * w


if IEEE_PICTURES:

    # plt.style.use(['science','ieee'])
    # plt.style.use(['science','ieee'])
    # plt.rcParams['text.usetex'] = False  # Desativa o uso de LaTeX externo para texto
    # plt.rcParams['legend.frameon'] = True  # Mostrar a moldura da legenda
    # plt.rcParams['legend.edgecolor'] = 'lightgray'  # Cor da borda da legenda
    # plt.rcParams['legend.facecolor'] = 'lightgray'  # Cor do fundo da legenda
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

    plt.style.use(['science','ieee'])
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = 'lightgray'
    plt.rcParams['legend.facecolor'] = 'lightgray'

    if ALL_IN_ONE:
        # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6.2, 5))
        # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6.2, 4.5), sharex=True, sharey=True)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6.2, 5.0), sharex=True, sharey=False)

        # VINCULAÇÃO PERSONALIZADA:
        # Força o gráfico (d) [índice 1,1] a ter o mesmo eixo Y do gráfico (a) [índice 0,0]
        axs[1, 1].sharey(axs[0, 0])

    else:
        fig_tdc, ax_tdc = plt.subplots(figsize=(10, 6), dpi=600)
        fig_dco, ax_dco = plt.subplots(figsize=(10, 6), dpi=600)
        fig_system, ax_system = plt.subplots(figsize=(10, 6), dpi=600)
        fig_system_iir, ax_system_iir = plt.subplots(figsize=(10, 6), dpi=600)

else:
    if ALL_IN_ONE:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    else:
        fig_tdc, ax_tdc = plt.subplots(figsize=(10, 6))
        fig_dco, ax_dco = plt.subplots(figsize=(10, 6))
        fig_system, ax_system = plt.subplots(figsize=(10, 6))
        fig_system_iir, ax_system_iir = plt.subplots(figsize=(10, 6))
    

for config in configs:
    alpha = config["alpha"]
    rho = config["rho"]
    l1 = irr_configs[0]["lambda1"]
    l2 = irr_configs[0]["lambda2"]
    l3 = irr_configs[0]["lambda3"]
    l4 = irr_configs[0]["lambda4"]
    qsi = 0.5 * alpha / (np.sqrt(rho))
    color_p = plot_color[configs.index(config)]["color_p"]
    
    # Open Loop Transfer Function
    HOL =  (alpha + (rho * fR)/s)* fR/s
    H_IIR1 = (1 + s/fR) / (1 + s/(fR * l1))
    H_IIR2 = (1 + s/fR) / (1 + s/(fR * l2))
    H_IIR3 = (1 + s/fR) / (1 + s/(fR * l3))
    H_IIR4 = (1 + s/fR) / (1 + s/(fR * l4))

    H_IIR = H_IIR1 * H_IIR2 * H_IIR3 * H_IIR4

    #closed loop transfer function TDC/FREF
    num_tdc = HOL
    den_tdc = 1 + HOL 
    H_cl_tdc =  (num_tdc / den_tdc) 

    #closed loop transfer function DCO
    num_dco = 1
    den_dco = 1 + HOL
    H_cl_dco = (num_dco / den_dco) 
    
  
    H_cl_dco_iir = H_cl_dco * H_IIR
    H_cl_tdc_iir = H_cl_tdc * H_IIR

    # Square Magnitude to linear scalar
    H_cl_tdc_mag2 = np.abs(H_cl_tdc)**2
    H_cl_dco_mag2 = np.abs(H_cl_dco)**2

    H_cl_tdc_iir_mag2 = np.abs(H_cl_tdc_iir)**2
    H_cl_dco_iir_mag2 = np.abs(H_cl_dco_iir)**2

    #System contribution to phase noise
    # 1. LINEAR POWER of each noise contribution at the output
    # TDC (Low-pass)
    pwr_tdc_out = (10**(S_tdc_floor / 10)) * H_cl_tdc_mag2
    pwr_tdc_out_iir = (10**(S_tdc_floor / 10)) * H_cl_tdc_iir_mag2
    
    # FREF (Low-pass) -> Multiplied by the squared loop gain (FCW**2)
    pwr_fref_out = (10**(-150 / 10)) * H_cl_tdc_mag2 * (FCW**2)
    pwr_fref_out_iir = (10**(-150 / 10)) * H_cl_tdc_iir_mag2 * (FCW**2)
    
    # DCO (High-pass)
    pwr_dco_out = (10**(S_dco_floor / 10)) * H_cl_dco_mag2
    pwr_dco_out_iir = (10**(S_dco_floor / 10)) * H_cl_dco_iir_mag2


    # 2. SUM OF POWERS IN LINEAR 
    total_noise_linear = pwr_tdc_out + pwr_fref_out + pwr_dco_out
    total_noise_linear_iir = pwr_tdc_out_iir + pwr_fref_out_iir + pwr_dco_out_iir

    # 3. FINAL CONVERSION TO DB
    phase_noise_tdc_dB = 10 * np.log10(pwr_tdc_out)
    phase_noise_dco_dB = 10 * np.log10(pwr_dco_out)
    H_cl_system = 10 * np.log10(total_noise_linear)
    H_cl_system_iir = 10 * np.log10(total_noise_linear_iir)
    
    if ALL_IN_ONE:
        axs[0, 0].semilogx(f_offset, phase_noise_tdc_dB, label=r"$\zeta$: " + f"{qsi:.2f}", color=color_p) #label=f"$\\alpha$={alpha:.4f}, $\\rho$={rho:.6f}")
        axs[0, 1].semilogx(f_offset, phase_noise_dco_dB, label=r"$\zeta$: " + f"{qsi:.2f}", color=color_p) #label=f"$\\alpha$={alpha:.4f}, $\\rho$={rho:.6f}")
        axs[1, 0].semilogx(f_offset, H_cl_system, label=r"$\zeta$: " + f"{qsi:.2f}", color=color_p) # label=f"$\\alpha$={alpha:.4f}, $\\rho$={rho:.6f}")
        axs[1, 1].semilogx(f_offset, H_cl_system_iir, label=r"$\zeta$: " + f"{qsi:.2f}", color=color_p) # label=f"$\\alpha$={alpha:.4f}, $\\rho$={rho:.6f}")
    else:
        ax_tdc.semilogx(f_offset, phase_noise_tdc_dB, label=r"$\zeta$: " + f"{qsi:.2f}", color=color_p) #label=f"$\\alpha$={alpha:.4f}, $\\rho$={rho:.6f}")
        ax_dco.semilogx(f_offset, phase_noise_dco_dB, label=r"$\zeta$: " + f"{qsi:.2f}", color=color_p) #label=f"$\\alpha$={alpha:.4f}, $\\rho$={rho:.6f}")
        ax_system.semilogx(f_offset, H_cl_system, label=r"$\zeta$: " + f"{qsi:.2f}", color=color_p) # label=f"$\\alpha$={alpha:.4f}, $\\rho$={rho:.6f}")
        ax_system_iir.semilogx(f_offset, H_cl_system_iir, label=r"$\zeta$: " + f"{qsi:.2f}", color=color_p) # label=f"$\\alpha$={alpha:.4f}, $\\rho$={rho:.6f}")

    


if ALL_IN_ONE:

    linhas_legenda = axs[0, 0].get_lines()
    labels_legenda = [l.get_label() for l in linhas_legenda]

    FONT_LABEL = 9
    FONT_TICKS = 8
    # --- Configurações do Gráfico TDC ---
    axs[0, 0].set_title("(a) FREF/TDC Transfer Function", fontsize=FONT_LABEL, fontweight='bold', pad=4)
    # axs[0, 0].set_xlabel("Frequency offset (Hz)", fontsize=12)
    # axs[0, 0].set_ylabel("Phase noise (dBc/Hz)", fontsize=12)
    # axs[0, 0].grid(True, which="minor", ls="--", alpha=0.5)
    # axs[0, 0].tick_params(axis='x', labelsize=FONT_TICKS)
    # axs[0, 0].tick_params(axis='y', labelsize=FONT_TICKS)
    # plt.yticks(fontsize=12)
    # axs[0, 0].legend(facecolor='white', framealpha=1, fontsize=FONT_LABEL)
    # plt.tight_layout()

    # --- Configurações do Gráfico DCO ---
    axs[0, 1].set_title("(b) DCO Transfer Function", fontsize=FONT_LABEL, fontweight='bold', pad=4)
    # axs[0, 1].set_xlabel("Frequency offset (Hz)", fontsize=12)
    # axs[0, 1].set_ylabel("Phase noise (dBc/Hz)", fontsize=FONT_LABEL)
    # axs[0, 1].grid(True, which="minor", ls="--", alpha=0.5)
    # axs[0, 1].tick_params(axis='x', labelsize=FONT_TICKS)
    # axs[0, 1].tick_params(axis='y', labelsize=FONT_TICKS)
    # axs[0, 1].legend(facecolor='white', framealpha=1, fontsize=FONT_LABEL)
    # plt.tight_layout()

    #config system grafics
    axs[1, 0].set_title("(c) System Transfer Function", fontsize=FONT_LABEL, fontweight='bold', pad=4)
    # axs[1, 0].set_xlabel("Frequency offset (Hz)", fontsize=FONT_LABEL)  
    # axs[1, 0].set_ylabel("Phase noise (dBc/Hz)", fontsize=FONT_LABEL)
    # axs[1, 0].grid(True, which="minor", ls="--", alpha=0.5)
    # axs[1, 0].tick_params(axis='x', labelsize=FONT_TICKS)
    # axs[1, 0].tick_params(axis='y', labelsize=FONT_TICKS)
    # axs[1, 0].legend(facecolor='white', framealpha=1, fontsize=FONT_LABEL)
    # plt.tight_layout()

    #config system_iir grafics
    axs[1, 1].set_title("(d) System Transfer Function + IIR", fontsize=FONT_LABEL, fontweight='bold', pad=4)
    # axs[1, 1].set_xlabel("Frequency offset (Hz)", fontsize=FONT_LABEL)  
    # axs[1, 1].set_ylabel("Phase noise (dBc/Hz)", fontsize=FONT_LABEL)
    # axs[1, 1].grid(True, which="minor", ls="--", alpha=0.5)
    # axs[1, 1].tick_params(axis='x', labelsize=FONT_TICKS)
    # axs[1, 1].tick_params(axis='y', labelsize=FONT_TICKS)
    # axs[1, 1].legend(facecolor='white', framealpha=1, fontsize=FONT_LABEL)
    # plt.tight_layout()
    for ax in axs.flat:
        ax.grid(True, which="minor", ls="--", alpha=0.3)
        ax.tick_params(axis='both', labelsize=FONT_TICKS)
        ax.tick_params(labelleft=True) 
        # ax.set_ylim(-180, -90)
    
    fig.supxlabel('Frequency offset (Hz)', fontsize=10)
    fig.supylabel('Phase noise (dBc/Hz)', fontsize=10) # Ou a unidade correta do seu Step

    fig.legend(handles=linhas_legenda, labels=labels_legenda, 
               loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=len(configs), 
               facecolor='white', framealpha=1, edgecolor='darkgrey', fontsize=FONT_LABEL)
    
    # 3. Organiza o layout da matriz 2x2
    fig.set_layout_engine('constrained')


else:

    # --- Configurações do Gráfico TDC ---
    ax_tdc.set_title("FREF/TDC Transfer Function", fontsize=12)
    ax_tdc.set_xlabel("Frequency offset (Hz)", fontsize=12)
    ax_tdc.set_ylabel("Phase noise (dBc/Hz)", fontsize=12)
    ax_tdc.grid(True, which="both", ls="--", alpha=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax_tdc.legend(facecolor='white', framealpha=1, fontsize=12)
    fig_tdc.tight_layout()

    # --- Configurações do Gráfico DCO ---
    ax_dco.set_title("DCO Transfer Function", fontsize=12)
    ax_dco.set_xlabel("Frequency offset (Hz)", fontsize=12)
    ax_dco.set_ylabel("Phase noise (dBc/Hz)", fontsize=12)
    ax_dco.grid(True, which="both", ls="--", alpha=0.5)
    # ax_dco.xticks(fontsize=12)
    # ax_dco.yticks(fontsize=12)
    ax_dco.legend(facecolor='white', framealpha=1, fontsize=12)
    fig_dco.tight_layout()

    #config system grafics
    ax_system.set_title("System Closed Loop Transfer Function", fontsize=12)
    ax_system.set_xlabel("Frequency offset (Hz)", fontsize=12)  
    ax_system.set_ylabel("Phase noise (dBc/Hz)", fontsize=12)
    ax_system.grid(True, which="both", ls="--", alpha=0.5)
    # ax_system.xticks(fontsize=12)
    # ax_system.yticks(fontsize=12)
    ax_system.legend(facecolor='white', framealpha=1, fontsize=12)
    fig_system.tight_layout()

    #config system_iir grafics
    ax_system_iir.set_title("System Closed Loop Transfer Function + IIR", fontsize=12)
    ax_system_iir.set_xlabel("Frequency offset (Hz)", fontsize=12)  
    ax_system_iir.set_ylabel("Phase noise (dBc/Hz)", fontsize=12)
    ax_system_iir.grid(True, which="both", ls="--", alpha=0.5)
    # ax_system_iir.xticks(fontsize=12)
    # ax_system_iir.yticks(fontsize=12)
    ax_system_iir.legend(facecolor='white', framealpha=1, fontsize=12)
    fig_system_iir.tight_layout()

################################################################################################################


if IEEE_PICTURES:
    if ALL_IN_ONE:
        fig.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\all_in_one_phase_noise_S_domain.pdf', bbox_inches='tight', format='pdf')
    else:
        fig_system.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\system_phase_noise_S_domain.pdf', bbox_inches='tight', format='pdf')
        fig_dco.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\dco_phase_noise_S_domain.pdf', bbox_inches='tight', format='pdf')
        fig_tdc.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\tdc_phase_noise_S_domain.pdf', bbox_inches='tight', format='pdf')
        fig_system_iir.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\system_phase_noise_S_domain_iir.pdf', bbox_inches='tight', format='pdf')

else:
    plt.show()