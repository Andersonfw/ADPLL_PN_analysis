from pathlib import Path as path
from pickle import TRUE

from matplotlib.rcsetup import cycler
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
IEEE_PICTURES = true


fref_predict_path = path("data/fref_predict_file.csv")
tdc_path = path("data/tdc_file.csv")

tdc = pd.read_csv(tdc_path, sep=';', header=None)
fref = pd.read_csv(fref_predict_path, sep=';', header=None)

time_fref_pre = fref[0].to_numpy()
ckv_gating = fref[1].to_numpy()
ckv = fref[2].to_numpy()
ckvd4 = fref[3].to_numpy()
fref_signal = fref[4].to_numpy()

time_tdc = tdc[0].to_numpy()
tdc_q = tdc[1].to_numpy()
fref_signal_tdc = tdc[2].to_numpy()
tdc_rise = tdc[3].to_numpy()
tdc_fall = tdc[4].to_numpy()
sel_edge = tdc[5].to_numpy()
tdc_q_sel_edge = tdc[6].to_numpy()

# Cada sinal digital varia de 0 a 1. Vamos dar um espaço vertical de 1.5 para cada um.
espaco = 1.5

if IEEE_PICTURES:
        plt.style.use(['science','ieee'])
        plt.style.use(['science','ieee'])
        plt.rcParams['legend.frameon'] = True  # Mostrar a moldura da legenda
        # plt.rcParams['legend.edgecolor'] = 'lightgray'  # Cor da borda da legenda
        # plt.rcParams['legend.facecolor'] = 'lightgray'  # Cor do fundo da legenda

        # --- CORREÇÃO AQUI: Força o estilo de linha a ser sempre sólido '-' ---
        # Mantém as cores padrões do tema IEEE, mas remove a alternância de traços
        plt.rcParams['axes.prop_cycle'] = cycler(color=['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747']) + cycler(linestyle=['-', '-', '-', '-', '-', '-'])

        # Suas configurações de legenda (ajustadas para melhor leitura no fundo cinza)
        plt.rcParams['legend.frameon'] = True  
        plt.rcParams['legend.edgecolor'] = 'lightgray'  
        plt.rcParams['legend.facecolor'] = '#F5F5F5'  # Um cinza bem claro para o texto preto contrastar bem

        plt.figure(figsize=(10, 5),dpi=600)

else:
        plt.figure(figsize=(10, 5))

# O truque está aqui: usar 'step' com where='post'
plt.step(time_fref_pre * 1e6, ckv_gating     * 1.0 + (0 * espaco)   , where='post', color='blue'  , linewidth=1.5)
plt.step(time_fref_pre * 1e6, ckvd4          * 1.0 + (1 * espaco)   , where='post', color='green' , linewidth=1.5)
plt.step(time_fref_pre * 1e6, fref_signal    * 1.0 + (2 * espaco)   , where='post', color='red'   , linewidth=1.5)
plt.step(time_fref_pre * 1e6, ckv            * 1.0 + (3 * espaco)   , where='post', color='orange', linewidth=1.5)

# Opcional: plota os pontos exatos onde o Verilog salvou o dado
posicoes_y = [0.5, 1.5 + 0.5, 3.0 + 0.5, 4.5 + 0.5]
nomes_sinais = ['CKV_GATING', 'CKVD4', 'FREF', 'CKV', ]

plt.yticks(posicoes_y, nomes_sinais, fontsize=12)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.ylim(-0.5, (3 * espaco) + 1.5)
plt.gca().xaxis.set_major_formatter(plt.NullFormatter())
# plt.xticks([]) # Passa uma lista vazia para remover os marcadores e números
# plt.xlabel('') # Remove o texto "Tempo (s)" se desejar
# plt.show()
if IEEE_PICTURES:
        plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\fref_predic_sim.pdf', bbox_inches='tight', format='pdf')

#########################################################################################################
# 1. Simulação dos dados de uma única linha do seu CSV para demonstração
# Na prática, você lerá do arquivo e pegará a linha desejada.
# Exemplo: tdc_q = 4380870835712 (em decimal)
tdc_q_decimal = tdc_q[0]
tdc_rise_val = tdc_rise[0]   # O valor que seu decoder achou para a subida
tdc_fall_val = tdc_fall[0]  # O valor que seu decoder achou para a descida

# Converte o número decimal gigante para um vetor binário de 42 bits
# '[2:]' remove o prefixo '0b' e '.zfill(42)' garante os 42 bits preenchidos
bin_string = bin(tdc_q_decimal)[2:].zfill(42)
# Convertemos a string de caracteres em uma lista de inteiros (0s e 1s)
# Invertemos a ordem se o bit [1] for o LSB ou MSB (ajuste conforme seu RTL)
tdc_bits = np.array([int(b) for b in bin_string]) 

# Redimensiona para uma matriz 1x42 para plotar como imagem/heatmap
matriz_tdc = tdc_bits.reshape(1, -1)

# 2. Construção do Gráfico
if IEEE_PICTURES:
        fig, ax = plt.subplots(figsize=(10, 3), dpi=600)
else:
        fig, ax = plt.subplots(figsize=(10, 3))

# Plota os 42 bits como quadradinhos (0 = azul escuro, 1 = amarelo/branco)
im = ax.imshow(matriz_tdc, cmap='binary', aspect='auto', extent=[0.5, 42.5, 0, 1])

# Adiciona linhas de grade para separar visualmente cada bit da Delay Line
ax.set_xticks(np.arange(0.5, 42.5, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

# Configuração dos eixos
ax.set_xticks(np.arange(1, 43, 2)) # Amostra os números de 2 em 2 bits
ax.tick_params(axis='x', labelsize=12)
ax.set_yticks([]) # Oculta o eixo Y já que é só uma linha temporária
ax.set_xlabel('Delay Line (Bits 1 to 42)', fontsize=12)
# ax.set_title(f'Estado Instantâneo do TDC e Saída do Decoder', fontsize=12, pad=15)

# 3. Plotar os Marcadores do Decoder (Onde ele achou as bordas)
# Colocamos uma seta ou marcador indicando as posições de subida e descida
ax.plot(tdc_rise_val, 0.5, marker='^', color='red', markersize=12, 
        label=f'TDC RISE (Bit {tdc_rise_val})')
ax.plot(tdc_fall_val, 0.5, marker='v', color='darkorange', markersize=12, 
        label=f'TDC FALL (Bit {tdc_fall_val})')

# Adiciona o valor numérico (0 ou 1) dentro de cada quadradinho para ficar ultra claro
for i in range(42):
    ax.text(i + 1, 0.5, str(tdc_bits[i]), ha='center', va='center', 
            color='black' if tdc_bits[i] == 0 else 'white', fontweight='bold', fontsize=12)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=True, fontsize=12)
plt.tight_layout()



if IEEE_PICTURES:
        fig.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\tdc_decoder_sim.pdf', bbox_inches='tight', format='pdf')
       
else:
    plt.show()




