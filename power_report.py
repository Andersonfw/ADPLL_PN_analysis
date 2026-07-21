import os
import re
import glob
from pathlib import Path
import pandas as pd

# 1. Configura os caminhos (Altere se a sua pasta POWER estiver em outro local)
power_dir = Path("data/TYP/POWER") 
OUTPUT_PATH = Path("output") # Caminho onde será salvo o relatório final
output_csv = Path(OUTPUT_PATH/"power_report.csv") # Caminho do arquivo final de saída

os.makedirs(OUTPUT_PATH, exist_ok=True)
# 2. Expressão regular para capturar o "Total Power" dentro do arquivo de texto
# Captura "Total Power:" seguido por espaços e qualquer número com pontos decimais
power_regex = re.compile(r"Total\s+Power:\s+([0-9\.]+)")

# 3. Lista para acumular as linhas da tabela final
dados_completos = []

# 4. Varre todos os relatórios .txt dentro da pasta POWER
relatorios = power_dir.glob("report_power_*.txt")

print(f"[PYTHON] Iniciando leitura dos relatórios em: {power_dir}")

for filepath in relatorios:
    filename = filepath.name  # Ex: report_power_2440_TYP_FREF_SDM1.txt
    
    # Divide o nome do arquivo pelo caractere '_' para extrair as variáveis
    # Removendo o prefixo 'report_power_' e o sufixo '.txt'
    parts = filename.replace("report_power_", "").replace(".txt", "").split("_")
    
    # Valida se o formato do nome tem as partes necessárias
    if len(parts) >= 3:
        freq_str = parts[0]
        if len(freq_str) > 4:
            freq_formatada = f"{freq_str[:4]}.{freq_str[4:]}"  # Vira "2418.123"
            freq = float(freq_formatada)
        else:
            freq = float(freq_str)
        corner = parts[1]
        # Junta o resto das partes caso o setting tenha underlines (ex: FREF_SDM1)
        setting = "_".join(parts[2:])
    else:
        print(f"[WARNING] Nome de arquivo fora do padrão: {filename}")
        continue
        
    # Abre o arquivo de texto para buscar o valor de consumo
    power_value = None
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = power_regex.search(line)
            if match:
                power_value = match.group(1).replace('.', ',') 
                break # Encontrou o Total Power, pode fechar o arquivo atual
                
    if power_value is not None:
        # Adiciona o dicionário mapeado na nossa lista de dados
        dados_completos.append({
            "FREQ": freq,
            "SETTING": setting,
            "CORNER": corner,
            "POWER (mW)": power_value
        })
    else:
        print(f"[WARNING] 'Total Power' não localizado no arquivo: {filename}")

# 5. Se coletou dados, cria o DataFrame do Pandas e exporta para CSV
if dados_completos:
    df = pd.DataFrame(dados_completos)
    
    # Ordena as linhas para a tabela ficar bonita (por Freq, depois Corner, depois Modo)
    df = df.sort_values(by=["FREQ", "CORNER", "SETTING"])
    
    # Salva o arquivo CSV separado por ponto e vírgula (;) conforme solicitado
    df.to_csv(output_csv, sep=";", decimal=",", index=False)
    print(f"\n[SUCESSO] Tabela gerada com {len(df)} registros!")
    print(f"Salvo em: {output_csv.resolve()}")
else:
    print("\n[ERRO] Nenhum dado de potência pôde ser extraído. Verifique os arquivos.")
