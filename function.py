""""
@author: Ânderson Felipe Weschenfelder
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class Colors:
    WHITE = '\033[97m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    YELLOW = '\033[0;33m'
    BLUE = "\033[0;34m"


def fun_calc_psd(x , fs=1 , rbw=100e3 , fstep=None):
    '''
    Calculate power spectral density

    INPUT arguments
    x     :  data vector
    fs    :  sample rate [Hz]
    rbw   :  resolution bandwidth [Hz]
    fstep :  FFT frequency bin separation [Hz]
    OUTPUT
    XdB	: spectrum of x [dB]
    f	: frequency vector [Hz]
    '''

    if fstep is None:
        fstep = rbw / 1.62
    len_x = len(x)
    nwin = round(fs * 1.62 / rbw)
    nfft = round(fs / fstep)
    if nwin > len_x:
        nwin = len_x
        rbw = fs * 1.62 / nwin
    num_segments = 8
    # nwin = math.floor(len(x) / num_segments)
    fftstr = (f'len(x)={len_x:.2f}, rbw={rbw / 1e3:.2f}kHz, fstep={fstep / 1e3:.2f}kHz, nfft={nfft:d}, nwin={nwin:d}')
    print(f'Calculating the PSD: {fftstr}')
    f , X = signal.welch(x , fs=fs , window=signal.windows.blackman(nwin) , nperseg=nwin , nfft=nfft ,scaling='density')
    X *= (np.sinc(f / fs)) ** 2  # correct for ZOH
    XdB = 10 * np.log10(X) - 3 # Subtrai 3dB para converter para L(f)
    XdB_sig = np.max(XdB) 
    print(f'Signal PSD peak = {XdB_sig:.2f} dB, 10log(rbw) = {10 * np.log10(rbw):.1f}')
    return XdB , f


def adpll_spectrum(phase, fs):
    """
    Calcula e plota o espectro para visualizar Espúrios e Ruído de Fase.
    """
    len_x = len(phase)
    T_sim_total = len_x / fs
    print(f"Tempo total de dados analisados: {T_sim_total * 1e6:.2f} us")
    
    # Se a simulação for menor que 20us, você terá dificuldade de ver spurs finos!
    if T_sim_total < 20e-6:
        print("ALERTA: Simulação muito curta para resolução fina de espúrios!")

    # -------------------------------------------------------------------
    # 1. ANÁLISE DE ESPÚRIOS (PERIODOGRAMA ÚNICO)
    # Pega TODO o vetor de uma vez, sem médias. Preserva a energia do pico.
    # -------------------------------------------------------------------
    f_spur, Pxx_spur = signal.periodogram(
        phase, 
        fs=fs, 
        window='hann',       # Hann é melhor que Blackman para separar tons
        nfft=len_x,          # FFT do tamanho exato dos dados (sem zero padding falso)
        scaling='spectrum'   # Escala de POTÊNCIA (para ler dBc direto no pico)
    )
    
    # O ruído de fase (PN) é a potência do erro de fase dividida por 2 (single sideband)
    # L(f) em dBc para os espúrios:
    L_spur_dBc = 10 * np.log10(Pxx_spur / 2)

    # -------------------------------------------------------------------
    # 2. ANÁLISE DE RUÍDO DE FASE (WELCH TRADICIONAL)
    # Para ver o "chão" de ruído limpo (como no seu código original)
    # -------------------------------------------------------------------
    nwin = len_x // 8 # 8 médias para limpar o ruído
    f_pn, Pxx_pn = signal.welch(
        phase, 
        fs=fs, 
        window='hann', 
        nperseg=nwin, 
        scaling='density' # Escala de DENSIDADE (para ler dBc/Hz)
    )
    L_pn_dBc_Hz = 10 * np.log10(Pxx_pn / 2)
# --- PLOT ---
    plt.figure(figsize=(10, 6))
    
    # Plota a Densidade (Noise Floor suave)
    plt.semilogx(f_pn, L_pn_dBc_Hz, label='Noise Floor (Welch - dBc/Hz)', color='C0', alpha=0.8)
    
    # Plota os Espúrios (Picos afiados)
    # Adicionamos um offset artificial (ex: - 10*log10(RBW)) só para alinhar visualmente 
    # o fundo do periodograma com o Welch, mas os PICOS estarão na amplitude certa.
    rbw_periodogram = fs / len_x
    plt.semilogx(f_spur, L_spur_dBc - 10*np.log10(rbw_periodogram), 
                 label=f'Espúrios (Periodograma)', color='red', alpha=0.5, linewidth=0.8)

    plt.title("Análise de Ruído de Fase e Espúrios do ADPLL")
    plt.xlabel("Frequency Offset (Hz)")
    plt.ylabel("Phase Noise [dBc/Hz] / Spur Power [dBc]")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlim([1e3, fs/2]) # Foca a partir de 10kHz até Nyquist
    plt.ylim([-160, -40])
    plt.legend()
    # plt.show()

    #return L_pn_dBc_Hz , f_pn
    return L_spur_dBc , f_spur

def edges_convert_to_freq_out_analyses(t_edges, window_time, time_cut_freq_analysis,silent=False):
    

    indice_edges_analyses = (np.abs(t_edges - time_cut_freq_analysis)).argmin()
    t_edges_cut = t_edges[indice_edges_analyses:] 
    N_edges = len(t_edges_cut)
    f_c = (N_edges - 1) / (t_edges_cut[-1] - t_edges_cut[0])

    t_periods = np.diff(t_edges_cut)
    f_inst = 1.0 / t_periods

    t_periods_full_analysis = np.diff(t_edges)
    f_inst__full_analysis = 1.0 / t_periods_full_analysis
    
    avg_period = np.mean(t_periods)
    window_size = int(window_time / avg_period)
    window = np.ones(window_size) / window_size
    f_smooth = np.convolve(f_inst, window, mode='valid')
    f_smooth_full_analysis = np.convolve(f_inst__full_analysis, window, mode='valid')


    if not silent:
        print(f"{Colors.GREEN}--- Análise de média movel com janela de {window_size} samples --- Window Time: {window_time * 1e6:.2f} us ---\r\n")
        print(f"{Colors.GREEN}Frequência Inicial Medida: {f_c/1e6:.4f} MHz ---\r\n")

        

    return {
        "f_c": f_c,
        "f_smooth": f_smooth,
        "f_inst": f_inst,
        "t_periods": t_periods,
        "index_cut_freq_analysis": indice_edges_analyses,
        "t_edges_cut": t_edges_cut,
        "f_smooth_full_analysis": f_smooth_full_analysis,
    }


# def edges_convert_to_freq_and_ble_validation(t_edges, time_cut_freq_analysis, silent=False):
#     """
#     Extrai frequência e valida métricas BLE (Drift e Drift Rate) a partir de bordas de clock.
#     """

#     BLE_WINDOW_TIME_ANALYSIS_DRIFT = 10e-6  # Janela oficial de 10 µs para suavização e análise de preâmbulo 10 symbols of 1 Mbps
#     BLE_WINDOW_TIME_ANALYSIS_DRIFT_RATE = 50e-6 # Janela oficial de 50 µs para análise de drift rate 
#     # 1. Cortar o tempo de acomodação (Settling Time) do ADPLL
#     indice_edges_analyses = (np.abs(t_edges - time_cut_freq_analysis)).argmin()
#     t_edges_cut = t_edges[indice_edges_analyses:] 
    
#     # Cria um eixo de tempo relativo (começando em 0) para a matemática das janelas
#     t_rel = t_edges_cut - t_edges_cut[0]

#     # 2. Extrair a Frequência Instantânea Bruta
#     t_periods = np.diff(t_edges_cut)
#     f_inst = 1.0 / t_periods
    
#     # Frequência central teórica média de todo o trecho cortado
#     f_c = (len(t_edges_cut) - 1) / (t_edges_cut[-1] - t_edges_cut[0])

#     # 3. Aplicar o Janelamento Oficial BLE (10 µs)
#     # Isso esmaga o ruído de quantização do SDM a 600 MHz
#     avg_period = np.mean(t_periods)
#     pontos_10us = int(BLE_WINDOW_TIME_ANALYSIS_DRIFT / avg_period)
    
#     window_ble = np.ones(pontos_10us) / pontos_10us
#     f_ble_smooth = np.convolve(f_inst, window_ble, mode='valid')
    
#     # Ajustar o eixo de tempo para alinhar com o sinal pós-convolução
#     t_ble_rel = t_rel[pontos_10us // 2 : -pontos_10us // 2 + 1]

#     # 4. Encontrar a Frequência de Referência f_0 (Preâmbulo: primeiros 10 µs)
#     idx_preamble_end = np.searchsorted(t_ble_rel, BLE_WINDOW_TIME_ANALYSIS_DRIFT)
    
#     # Proteção caso a simulação seja curta demais
#     if idx_preamble_end == 0 or len(t_ble_rel) <= idx_preamble_end:
#         if not silent:
#             print(f"{Colors.RED}Erro: A janela de simulação pós-lock é curta demais para medir o Preâmbulo (10 µs).")
#         return None

#     f_ref = np.mean(f_ble_smooth[:idx_preamble_end])

#     # 5. Calcular Max Drift Absoluto no Payload (Restante do pacote após 10 µs)
#     f_payload = f_ble_smooth[idx_preamble_end:]
#     drift_array = f_payload - f_ref
#     max_drift = np.max(np.abs(drift_array))

#     # 6. Calcular Max Drift Rate (Atraso oficial de 50 µs da spec de teste)
#     pontos_50us = int(BLE_WINDOW_TIME_ANALYSIS_DRIFT_RATE / avg_period)
    
#     if len(f_payload) > pontos_50us:
#         # Subtrai o array defasado de 50us do atual e divide por 50us
#         drift_rate_array = np.abs(f_payload[pontos_50us:] - f_payload[:-pontos_50us]) / 50.0 
#         max_drift_rate = np.max(drift_rate_array)
#         status_rate = "PASS" if max_drift_rate <= 400.0 else "FAIL"
#     else:
#         max_drift_rate = float('nan')
#         status_rate = "N/A"
#         if not silent:
#             print(f"{Colors.RED}Atenção: A simulação pós-lock precisa ser maior que 60 µs para avaliar a taxa com janela de 50 µs.")

#     status_drift = "PASS" if max_drift <= 50e3 else "FAIL"

#     # 7. Prints Formatados
#     if not silent:
#         # Assumindo que Colors já existe no seu script principal
#         try:
#             cor = Colors.GREEN
#             fim = getattr(Colors, 'ENDC', '\033[0m')
#         except NameError:
#             cor = ""
#             fim = ""
            
#         print(f"{cor}--- Análise de Conformidade BLE RF-PHY ---{fim}")
#         print(f"{cor}Sinal analisado após: {time_cut_freq_analysis * 1e6:.2f} µs (Lock Time){fim}")
#         print(f"{cor}Frequência de Referência (f_0): {f_ref/1e6:.4f} MHz{fim}")
#         print(f"{cor}Máximo Drift Medido: {max_drift/1e3:.2f} kHz  (Limite: ±50 kHz) -> [{status_drift}]{fim}")
        
#         if not np.isnan(max_drift_rate):
#             print(f"{cor}Máximo Drift Rate (∆t=50µs): {max_drift_rate:.2f} Hz/µs (Limite: 400 Hz/µs) -> [{status_rate}]{fim}\n")

#     # 8. Retorna todos os dados para plotagem ou processamento extra (como Phase Noise)
#     return {
#         "f_c": f_c,                           # Frequência teórica
#         "f_ref": f_ref,                       # Frequência base real medida (f_0)
#         "f_inst": f_inst,                     # Sinal bruto cheio de ruído
#         "t_periods": t_periods,               # Períodos crus
#         "t_edges_cut": t_edges_cut,           # Eixo X bruto
#         "f_ble_smooth": f_ble_smooth,         # Sinal filtrado pelo padrão de 10 µs (ótimo para plotar)
#         "t_ble_rel": t_ble_rel,               # Eixo X relativo filtrado (ótimo para plotar)
#         "drift_array": drift_array,           # Erro em relação a f_ref
#         "max_drift": max_drift,               # Valor consolidado em Hz
#         "max_drift_rate": max_drift_rate,     # Valor consolidado em Hz/us
#         "status_drift": status_drift,         # String "PASS" ou "FAIL"
#         "status_rate": status_rate            # String "PASS" ou "FAIL"
#     }

def edges_convert_to_freq_ble_compliance(t_edges, time_cut_freq_analysis, f_0, silent=False):
    """
    Analisa métricas BLE segmentando a frequência em blocos discretos de 10 µs.
    """
    BLE_WINDOW_TIME_ANALYSIS_DRIFT = 10e-6  # Janela oficial de 10 µs para suavização e análise de preâmbulo 10 symbols of 1 Mbps
    BLE_WINDOW_TIME_ANALYSIS_DRIFT_RATE = 50e-6 # Janela oficial de 50 µs para análise de drift rate 
    BLE_MAX_FREQUENCY_DRIFT_HZ = 50e3  # Limite de drift absoluto em Hz (±50 kHz)
    BLE_MAX_DRIFT_RATE_HZ_PER_US = 400.0 # Limite de drift rate em Hz/µs (400 Hz/µs)
    BLE_MAX_DRIFT_RATE_HZ_PER_US_IN_HZ = (BLE_MAX_DRIFT_RATE_HZ_PER_US * BLE_WINDOW_TIME_ANALYSIS_DRIFT_RATE) * 1e6 # Limite de drift rate em kHz
    BLE_MAX_FREQ_DEVIATION_HZ = 150e3 # Limite de desvio total permitido em relação a f_0 (±150 kHz)

    # 1. Cortar o tempo de acomodação (Settling Time) do ADPLL
    indice_edges_analyses = (np.abs(t_edges - time_cut_freq_analysis)).argmin()
    t_edges_cut = t_edges[indice_edges_analyses:] 
    
    # 2. Extrair a Frequência Instantânea Bruta e a Frequência Central (f_c)
    t_periods = np.diff(t_edges_cut)
    f_inst = 1.0 / t_periods
    f_c = (len(t_edges_cut) - 1) / (t_edges_cut[-1] - t_edges_cut[0])

    # 3. Determinar o tamanho exato de um "Bloco" de 10 µs em amostras
    avg_period = np.mean(t_periods)
    amostras_por_bloco = int(BLE_WINDOW_TIME_ANALYSIS_DRIFT / avg_period)
    
    # Quantos blocos inteiros de 10 us conseguimos extrair da simulação?
    num_blocos = len(f_inst) // amostras_por_bloco
    
    if num_blocos < 2:
        print(f"{Colors.RED}Erro: A simulação não tem tamanho suficiente para formar blocos BLE.")
        return None

    # 4. Criar o array de frequências f_n (Média escalar de cada bloco de 10 us)
    f_n = np.zeros(num_blocos)
    for i in range(num_blocos):
        inicio = i * amostras_por_bloco
        fim = inicio + amostras_por_bloco
        f_n[i] = np.mean(f_inst[inicio:fim])  # A média elimina o ruído do SDM a 600 MHz

    # 5. O Preâmbulo (f_0) é simplesmente o primeiro bloco
    f_ref = f_n[0]
    f_ref_deviation = np.abs(f_ref - f_0)
    status_f_ref_deviation = "PASS" if f_ref_deviation <= BLE_MAX_FREQ_DEVIATION_HZ else "FAIL"


    # 6. Calcular o Max Drift Absoluto (Teste 1)
    # Compara todos os blocos subsequentes (payload) com o f_ref
    drift_absoluto_array = np.abs(f_n[1:] - f_ref)
    max_drift = np.max(drift_absoluto_array)
    status_drift = "PASS" if max_drift <= BLE_MAX_FREQUENCY_DRIFT_HZ else "FAIL"

    # 7. Calcular o Max Drift Rate (Teste 2)
    # Compara o bloco atual com o bloco de 5 posições atrás (5 * 10 us = 50 us)
    if num_blocos >= 6:  # Precisamos de f_0 até f_5 (ou seja, mínimo de 60 µs simulados pós-lock)
        
        # Matematicamente idêntico à norma: | f_n - f_{n-5} | <= 20 kHz
        drift_rate_deltas = np.abs(f_n[5:] - f_n[:-5])
        max_drift_rate_delta_hz = np.max(drift_rate_deltas)
        
        # Converte para Hz/us apenas para exibição
        max_drift_rate_hz_us = max_drift_rate_delta_hz / 50.0 
        
        status_rate = "PASS" if max_drift_rate_delta_hz <= BLE_MAX_DRIFT_RATE_HZ_PER_US_IN_HZ else "FAIL"
    else:
        max_drift_rate_delta_hz = float('nan')
        max_drift_rate_hz_us = float('nan')
        status_rate = "N/A"
        if not silent:
            print(f"{Colors.RED}Aviso: Simulação pós-lock é menor que 60 µs. Impossível calcular Drift Rate (f_n - f_n-5).")

    # 8. Prints
    if not silent:
        print(f"{Colors.BLUE}--- Relatório do Analisador BLE RF-PHY ---")
        print(f"{Colors.GREEN}Janela de análise: {num_blocos * 10} µs ({num_blocos} blocos de 10 µs analisados)")
        print(f"{Colors.GREEN}Frequência Central Teórica: {f_c/1e9:.6f} GHz")
        print(f"{Colors.GREEN}Frequência de Referência (f_0): {f_ref/1e9:.6f} GHz")

        print(f"{Colors.BLUE}\n--- 1. Max Frequency Deviation ---")
        color = Colors.GREEN if status_f_ref_deviation == "PASS" else Colors.RED
        print(f"{color}Desvio de Referência: {f_ref_deviation:.2f} Hz")
        print(f"{color}Limite: 150.00 kHz")
        print(f"{color}Status de Desvio de Referência: [{status_f_ref_deviation}]")

        print(f"{Colors.BLUE}\n--- 2. Max Frequency Drift ---")
        color = Colors.GREEN if status_drift == "PASS" else Colors.RED
        print(f"{color}Medido: {max_drift:.2f} Hz")
        print(f"{color}Limite: 50.00 kHz")
        print(f"{color}Status: [{status_drift}]")
        
        if num_blocos >= 6:
            print(f"{Colors.BLUE}\n--- 3. Max Frequency Drift Rate ---")
            color = Colors.GREEN if status_rate == "PASS" else Colors.RED
            print(f"{color}Variação Medida em 50 µs: {max_drift_rate_delta_hz:.2f} Hz (Equivale a {max_drift_rate_hz_us:.2f} Hz/µs)")
            print(f"{color}Limite: 20.00 kHz (Equivale a 400.00 Hz/µs)")
            print(f"{color}Status: [{status_rate}]\n")

    return {
        "f_c": f_c,
        "f_ref": f_ref,
        "blocos_f_n": f_n, # O array contendo a frequência média de cada bloco de 10us
        "max_drift": max_drift,
        "max_drift_rate_hz": max_drift_rate_delta_hz,
        "status_drift": status_drift,
        "status_rate": status_rate,
        "status_f_ref_deviation": status_f_ref_deviation,
        "f_ref_deviation_hz": f_ref_deviation
    }