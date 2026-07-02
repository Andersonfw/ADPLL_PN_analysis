import numpy as np
import matplotlib.pyplot as plt
import control
import sympy as sp
import warnings
from scipy import signal
import scienceplots

warnings.filterwarnings('ignore')

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
    print(f'Calculating the PSD: {fftstr} ...')
    f , X = signal.welch(x , fs=fs , window=signal.windows.blackman(nwin) , nperseg=nwin , nfft=nfft ,scaling='density')
    X *= (np.sinc(f / fs)) ** 2  # correct for ZOH
    XdB = 10 * np.log10(X)
    XdB_sig = np.max(XdB)
    print(f'Signal PSD peak = {XdB_sig:.2f} dB, 10log(rbw) = {10 * np.log10(rbw):.1f}')
    return XdB , f


'''
            Exibir duas funções de transferência em uma mesma figura
            Parameters
            ----------
            name1 : Nome da legenda da função 1
            name2 : Nome da legenda da função 2
            sys1 : Função de transferênci da função 1
            sys1 : Função de transferênci da função 2
            w :   Lista de frequências para ser usada pela resposta em frequência
            margem : (opcional) Plotar a margem e ganho de fase
'''
def format_plot(name1, name2, sys1, sys2,  omega, margins=False):
    mag1, phase1, omega1 = control.bode(sys1, omega, Hz=False, Plot=False)
    mag2, phase2, omega2 = control.bode(sys2, omega, Hz=False, Plot=False)

    mag_dB1 = 20 * np.log10(mag1)
    mag_dB2 = 20 * np.log10(mag2)
    if margins:
        gm1, pm1, sm1, wg1, wp1, ws1 = control.stability_margins(sys1)
        gm2, pm2, sm2, wg2, wp2, ws2 = control.stability_margins(sys2)
        wp1 = wp1 / (2 * np.pi)
        wg1 = wg1 / (2 * np.pi)
        wp2 = wp2 / (2 * np.pi)
        wg2 = wg2 / (2 * np.pi)

    omega1 = omega1 / (2 * np.pi)
    omega2 = omega2 / (2 * np.pi)


    plt.subplot(221)
    if margins:
        plt.hlines(0, omega1[0], omega1[-1], linestyle='--')
        plt.vlines([wp1, wg1], np.min(mag_dB1), np.max(mag_dB1), linestyle='--')
    plt.semilogx(omega1, mag_dB1, '-r',  label="{}".format(name1))
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid()

    if margins:
        plt.title(
            ' bode pm: {:0.2f} deg @{:0.2f} Hz gm: {:0.2f} dB @{:0.2f} Hz'.
            format(pm1, wp1,  20 * np.log10(gm1), wg1))
    else:
        plt.title(name1 + ' bode')
    plt.subplot(223)
    phase_deg = np.rad2deg(phase1)
    plt.semilogx(omega1, phase_deg, '-r',  label="{}".format(name1))
    if margins:
        plt.vlines([wp1, wg1],
                   np.min(phase_deg),
                   np.max(phase_deg),
                   linestyle='--')
        plt.hlines(-180, omega1[0], omega1[-1], linestyle='--')
    plt.legend()
    plt.grid()

    plt.subplot(222)
    if margins:
        plt.hlines(0, omega2[0], omega2[-1], linestyle='--')
        plt.vlines([wp2, wg2], np.min(mag_dB2), np.max(mag_dB2), linestyle='--')
    plt.semilogx(omega2, mag_dB2, '-b',  label="{}".format(name2))
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid()

    if margins:
        plt.title(
            ' bode pm: {:0.2f} deg @{:0.2f} Hz gm: {:0.2f} dB @{:0.2f} Hz'.
            format(pm2, wp2,  20 * np.log10(gm2), wg2))
    else:
        plt.title(name2 + ' bode')

    plt.subplot(224)
    phase_deg = np.rad2deg(phase2)
    plt.semilogx(omega2, phase_deg, '-b',  label="{}".format(name2))
    if margins:
        plt.vlines([wp2, wg2],
                   np.min(phase_deg),
                   np.max(phase_deg),
                   linestyle='--')
        plt.hlines(-180, omega2[0], omega2[-1], linestyle='--')
    plt.legend()
    plt.grid()
'''
            Extrai os parêmetros do numerador e denominador de uma função de transferência em S
            Parameters
            ----------
            H : Função de transferência no dominio de frequência S
'''
def extract_parameters(H):
    # Converter a string em uma expressão simbólica
    expressao = sp.sympify(H)
    # Obter os numeradores e denominadores como objetos simbólicos
    num_Hol, den_Hol = expressao.as_numer_denom()
    # print("numerador em s:", num_Hol)
    # print("denominador em s:", den_Hol)
    # Extrair coeficientes do numerador e denominador de Hs
    num_Hol = sp.Poly(num_Hol, s).all_coeffs()
    den_Hol = sp.Poly(den_Hol, s).all_coeffs()
    # print("numerador:", num_Hol)
    # print("denominador:", den_Hol)
    # Transforma em array para aplicar a transformada de laplace
    num = np.array(num_Hol, dtype=float)
    den = np.array(den_Hol, dtype=float)
    return num, den


'''
            Calcula o filtro IRR de acordo com os coeficientes recebidos
            Parameters
            ----------
            lp : array de coeficientes
'''


def IRR_filter(lp):
    IRR_order = len(lp)
    IRR = 1
    for i in range(IRR_order - 1):
        IRR *= (1 + s / fr) / (1 + (s / (lp[i] * fr)))
    IRR *= (1 + s / fr) / (1 + (s / (lp[IRR_order - 1] * fr)))

    return IRR


'''
            Plota o gráfico de Bode na função tranferência no dominío S 
            Parameters
            ----------
            H : (opcional) Função de transferência no dominío de frequência S
            numerador : (opcional) array dos parâmetros do numerador da Função de transferência sem "S"
            denominador : (opcional) array dos parâmetros do denominador da Função de transferência sem "S"
            irr : (opcional) array de coeficientes do filtro IRR
            w : (opcional) Lista de frequências para ser usada pela resposta em frequência
            margem : (opcional) Plotar a margem e ganho de fase
            prints : (opcional) Printar informações
'''


def h_to_bode(H=None, numerador=None, denominador=None, irr=None, freq=None, margem=False, prints=False, plot=False):
    if irr:
        IRR = IRR_filter(irr)
    else:
        IRR = 1
    if H:
        num_irr, den_irr = extract_parameters(IRR)
        IRR = control.tf(num_irr, den_irr)
        num_H, den_H = extract_parameters(H)
        tf = control.tf(num_H, den_H)
    else:
        num_irr, den_irr = extract_parameters(IRR)
        IRR = control.tf(num_irr, den_irr)
        tf = control.tf(numerador, denominador)
    if prints:
        print("Função de transfêrencia sem filtro IRR: ", tf)
    tf = tf * IRR
    # tf = (tf /(1 + tf))
    # tf = (1 / (1 + tf))
    # tf = control.feedback(tf, 1)
    if prints:
        print("função de transferencia em Laplace com filtro IRR: ", tf)
    if freq is not None:
        mag, phase, f = control.bode(tf, omega=freq, Hz=True, dB=True, deg=True, margins=margem, plot=plot)
    else:
        mag, phase, f = control.bode(tf, Hz=True, dB=True, deg=True, margins=margem, plot=plot)

    return mag, phase, f, tf


'''
        DEFINIÇÕES
'''
# Símbolo 's' para a variável de Laplace
s = sp.symbols('s')
N = 4.8e9/26e6  # relação de f/f_r

# Livro Bogdan PG 137/152
# fr = 26e6  # Frequeência de referência
# a = 2 ** -7  # alpha value
# p = 2 ** -15  # rho value
# #   Coeficiêntes do filtro IIR
# l = [2**-3, 2**-3, 2**-3, 2**-4]

# Aassignment 4
# fr = 40e6  # Frequeência de referência
fr = 26e6  # Frequeência de referência
# a = 2 ** -7  # alpha value
a = 2 ** -5  # alpha value
# p = 2 ** -14  # rho value
p = 2 ** -8  # rho value
# pa = [2 ** -10 , 2 ** -11 , 2 ** -12 , 2 ** -13, 2 ** -14]
pa = [2 ** -10 , 2 ** -11, 2 ** -12, 2 ** -13, 2 ** -14]
pas = [ '$2^{{-10}}$' , '$2^{{-11}}$' , '$2^{{-12}}$', '$2^{{-13}}$', '$2^{{-14}}$']
# pas = ['$2^{{-10}}$', '$2^{{-11}}$', '$2^{{-12}}$', '$2^{{-13}}$', '$2^{{-14}}$']  # Seu array de strings pas

plot_color = [
    {"color_p": "red"},
    {"color_p": "blue"},
    {"color_p": "black"},
    {"color_p": "green"},
    {"color_p": "orange"}
]

# Coeficiêntes do filtro IIR
# l = [2 ** -2, 2 ** -3, 2 ** -2, 2 ** -3]
l = [2 ** -2 , 2 ** -1 , 2 ** -1 , 2 ** -1]

# Open Loop Unit Gain
w1 = a * fr * ( 0.5 + 0.5 * np.sqrt(1 + (4 * p / a**2)))

print("Frequência de ganho unitarío é: ", w1)
# Função de tranferência de loop aberto
#HOL = (a + p * fr / s) * (fr / s)
# HOL = a * (fr/s)

if __name__ == "__main__":
    '''
    loop BandWidth = alpha/2pi * FREF
    
    '''
    # # Função de tranferência de loop fechado para o TDC
    w = np.logspace(3, 8, 10000)  # List of frequencies in rad/sec to be used for frequency response ( 10^-1 até 10^3)
    margem = True
    # plt.style.use(['science','ieee'])
    # plt.style.use(['science','ieee'])
    # plt.rcParams['legend.frameon'] = True  # Mostrar a moldura da legenda
    # plt.rcParams['legend.edgecolor'] = 'lightgray'  # Cor da borda da legenda
    # plt.rcParams['legend.facecolor'] = 'lightgray'  # Cor do fundo da legenda
    # plt.figure(figsize=(3.54,2), dpi=600)
    # plt.figure(figsize=(3.54,2.5), dpi=600)
    plt.style.use(['science','ieee'])
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = 'lightgray'
    plt.rcParams['legend.facecolor'] = 'lightgray'
    plt.figure(figsize=(8, 5),dpi=600)
    # Plotar a resposta ao degrau
    #plt.plot(t, y)
    # 1. Crie um vetor de tempo adequado (ex: de 0 a 20 microssegundos, com 1000 pontos)
    t_vector = np.linspace(0, 20e-6, 1000)
    for sys in range(len(pa)):
        color_p = plot_color[sys]["color_p"]
        qsi = 0.5 * a / (np.sqrt(pa[sys]))
        HOL = (a + pa[sys] * fr / s) * (fr / s)
        mag, phase, f, Hol = h_to_bode(H=HOL, freq=w, irr=None, margem=margem, prints=True, plot=False)
        Hcl_TDC = (Hol / (1 + Hol))
        t, y = control.step_response(Hcl_TDC, T=t_vector)
        plt.plot(t, y, label=r"$\zeta$: "+ f"{qsi:.2f}  $K_I$={pas[sys]}", color=color_p)
    #plt.title('Resposta ao Degrau da função $H_{TDC}$ variando $\zeta$ com $K_p=2^-5$ e $f_{REF}=$ 26MHz')
    # plt.xlabel('Time (s)')#, fontsize=13)
    plt.xlabel('Time (s)', fontsize=17)
    # plt.ylabel('Step responde')#,fontsize=13)
    plt.ylabel('Step Response',fontsize=17)
    plt.legend(facecolor='white', framealpha=1,fontsize=13)
    plt.xlim(0, 2e-5)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.grid()
    # plt.savefig(r'C:\Users\ander\OneDrive\Área de Trabalho\artigo\step_htdc.eps', bbox_inches='tight', format='eps')

    plt.tight_layout()
    # plt.show()
    plt.savefig(r'C:\Users\ander\OneDrive - Associacao Antonio Vieira\Mestrado\ADPLL_PN_analysis\img\hcl_tdc_qsi_step_response.pdf', bbox_inches='tight', format='pdf')
