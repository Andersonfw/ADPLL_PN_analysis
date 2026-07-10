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

# Importa a função do arquivo secundário que criamos no Passo 1
import process as pr
import utilities as ut


window_time = 0.5e-6  # Tamanho da janela para suavização (1us para BLE)
time_cut_PN_start = 1.8e-04
time_cut_plot_start = time_cut_PN_start#5.0e-4 #1.2e-04
time_cut_plot_stop = 250e-06



# ==============================================================================
# CONFIGURAÇÃO DE CAMINHOS GLOBAIS
# ==============================================================================
RAIZ_DATA = path("data/MA") # Caminho base onde estão as subpastas
OUTPUT_PATH = path("output") # Caminho onde será salvo o relatório final
CSV_SAIDA = path(OUTPUT_PATH/"analyses_result.csv") # Caminho do arquivo final de saída

os.makedirs(OUTPUT_PATH, exist_ok=True)
# Lista para acumular a linha de dados de cada uma das subpastas
linhas_relatorio = []

print(f"[MAIN] Iniciando varredura na pasta: {RAIZ_DATA.resolve()}")

# 1. Varre o primeiro nível de diretórios (Ex: 2402_TYP, 2440_WORST)
for pasta_corner_freq in RAIZ_DATA.iterdir():
    if not pasta_corner_freq.is_dir():
        continue # Ignora arquivos soltos na raiz como o histórico CSV
        
    # Extrai Frequência e Corner do nome da pasta (Ex: "2402_TYP" -> ["2402", "TYP"])
    partes_nome = pasta_corner_freq.name.split("_")
    if len(partes_nome) < 2:
        continue # Ignora pastas fora do padrão de nome do script Bash
        
    freq_str = partes_nome[0]  # Converte a frequência para inteiro (Ex: "2402" -> 2402)
    if len(freq_str) > 4:
        freq_formatada = f"{freq_str[:4]}.{freq_str[4:]}"  # Vira "2418.123"
        freq_atual = float(freq_formatada)
    else:
        freq_atual = float(freq_str)
    corner_atual = partes_nome[1]

    # 2. Varre o segundo nível de diretórios (Ex: BASE_DUMPVARS, FREF_DUMPVARS)
    for pasta_modo in pasta_corner_freq.iterdir():
        if not pasta_modo.is_dir():
            continue
            
        data_path = str(pasta_modo)
        
        # Extrai o nome do modo limpo para usar depois no relatório (Ex: "FREF")
        if "_DUMPVARS" in pasta_modo.name:
            continue # Ignora pastas de dumpvars, pois não são modos válidos
        folder_name = pasta_modo.name
        
        # --- load Files ---
        fsm_path      = ut.get_latest_file(data_path, "fsm_states", "csv")
        t_edges_path  = ut.get_latest_file(data_path, "close_loop_edge_times", "txt")
        bank_path     = ut.get_latest_file(data_path, "bank_cap", "csv")
        phe_path      = ut.get_latest_file(data_path, "phe", "csv")
        otw_path      = ut.get_latest_file(data_path, "otw", "csv")
        active_settings_path = ut.get_latest_file(RAIZ_DATA, "sim_historic", "csv")

        fsm_file = pd.read_csv(fsm_path, sep=';', header=None)
        


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


        result = pr.process(t_edges, 
            window_time, 
            time_cut_plot_start, 
            time_cut_plot_stop, 
            freq_atual*1e6, 
            time_cut_PN_start, 
            fsm_file, 
            bank_files, 
            phe, 
            otw, 
            i_stop_ckv, 
            i_stop_banks, 
            plot_all=false, 
            IEEE_en=false, 
            PN_analysis=true)
        result["FREQ"] = freq_atual
        result["CORNER"] = corner_atual
        result["SETTING"] = folder_name
        linhas_relatorio.append(result)
        print(f" -> Processado com sucesso: Freq={freq_atual} | Corner={corner_atual} | Modo={folder_name}")

# ==============================================================================
# EXPORTAÇÃO DOS RESULTADOS TOTAIS
# ==============================================================================
if linhas_relatorio:
    df_final = pd.DataFrame(linhas_relatorio)
    
    # Organiza as linhas por frequência e corner para o relatório ficar perfeito
    df_final = df_final.sort_values(by=["FREQ", "CORNER", "SETTING"])
    
    # Exporta aplicando os separadores regionais (; para colunas e , para decimais)
    df_final.to_csv(CSV_SAIDA, sep=";", decimal=",", index=False)
    
    print(f"\n[SUCESSO] Varredura Concluída! Relatório gerado com {len(df_final)} cenários.")
    print(f"Salvo em: {CSV_SAIDA.resolve()}")
else:
    print("\n[ERRO] Nenhuma subpasta válida contendo simulações pôde ser processada.")
