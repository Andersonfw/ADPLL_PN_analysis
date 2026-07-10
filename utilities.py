from pathlib import Path
import glob
import os
import pandas as pd
import numpy as np


def get_latest_file(path, base_name, extension):
    # Procura por qualquer arquivo que comece com o nome e termine com a extensão na pasta 'data/'
    # O '*' captura qualquer timestamp (ex: data/phe_20260708_1715.csv)
    pattern = f"{path}/{base_name}*.{extension}"
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado para o padrão: {pattern}")
    
    # Retorna o arquivo com a maior data de modificação recente do sistema operacional
    latest_file = max(files, key=os.path.getmtime)
    return Path(latest_file)

class Colors:
    WHITE = '\033[97m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    YELLOW = '\033[0;33m'
    BLUE = "\033[0;34m"
