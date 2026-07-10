from pathlib import Path
import pandas as pd

# 1. Definição dos caminhos dos arquivos
csv_power = Path("output/power_report.csv") # ou o seu caminho do power
csv_analysis = Path("output/analyses_result.csv")
csv_final_unificado = Path("output/final_results.csv")

print("[PYTHON] Iniciando a fusão dos relatórios...")

print("[PYTHON] Adicionando a coluna de POWER ao relatório de análises...")

# 2. Carrega as tabelas como string
df_analysis = pd.read_csv(csv_analysis, sep=";", dtype=str)
df_power    = pd.read_csv(csv_power, sep=";", dtype=str)

# ------------------------------------------------------------------
# 3. LIMPEZA E PADRONIZAÇÃO (Garante o casamento perfeito das chaves)
# ------------------------------------------------------------------
for df in [df_analysis, df_power]:
    # Remove espaços em branco invisíveis no início ou fim dos textos
    df["FREQ"] = df["FREQ"].str.strip()
    df["CORNER"] = df["CORNER"].str.strip()
    df["SETTING"] = df["SETTING"].str.strip()
    
    # Padroniza a frequência removendo pontos ou vírgulas se houver
    # Ex: transforma "2418.123" ou "2418,123" em "2418123" temporariamente para cruzar
    df["FREQ_CHAVE"] = df["FREQ"].str.replace(".", "", regex=False).str.replace(",", "", regex=False)

# 4. Isola as colunas necessárias do arquivo de potência usando a nova FREQ_CHAVE
df_power_filtrado = df_power[["FREQ_CHAVE", "CORNER", "SETTING", "POWER (mW)"]]

# Remove duplicatas se houver mais de uma linha idêntica no arquivo de potência
df_power_filtrado = df_power_filtrado.drop_duplicates(subset=["FREQ_CHAVE", "CORNER", "SETTING"])

# 5. Faz o Merge usando a FREQ_CHAVE padronizada
df_final = pd.merge(
    df_analysis, 
    df_power_filtrado, 
    on=["FREQ_CHAVE", "CORNER", "SETTING"], 
    how="left"
)

# 6. Remove a coluna temporária de chave antes de salvar
df_final = df_final.drop(columns=["FREQ_CHAVE"])

# 5. Sobrescreve o seu arquivo original com a nova coluna inclusa
df_final.to_csv(csv_final_unificado, sep=";", decimal=",", index=False)

print(f"\n[SUCESSO] Coluna 'POWER' anexada com sucesso ao arquivo: {csv_final_unificado.resolve()}")