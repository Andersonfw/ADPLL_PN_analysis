import numpy as np



FREF = 26e6

rho = 2**(-10)
alpha = 2**(-4)


qsi = 0.5 * alpha / np.sqrt(rho)

if qsi > 0.5:
    bw = FREF * alpha
else:
    bw = FREF * (alpha  / (qsi*2))



print(f"ALPHA = {alpha:.6f}")
print(f"RHO = {rho:.6f}")


print(f"QSI = {qsi:.3f}")
print(f"BW = {bw/(2*np.pi)/1e6:.3f} MHz")
