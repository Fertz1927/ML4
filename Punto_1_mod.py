import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pywt
import glob



# ================================================================
# CONFIGURACIÓN
# ================================================================

GPS_START = 1239082260
GPS_END   = 1239082264
GPS_EVENT = 1239082262.2
DETECTOR  = "H"
DATASET   = "GW190412"

ENERGY = 0.95        # 95 % de la energía
WAVELET = "db4"        # familia ortogonal
LEVEL = 5              # nivel exigido por el taller


#================================================
# LOCALIZAR ARCHIVO DE DATOS
#================================================

filepath = "H-H1_GWOSC_4KHZ_R1-1239082247-32.hdf5"

if not os.path.exists(filepath):
    raise FileNotFoundError(
        f"No se encontró el archivo: {filepath}"
    )

print("Usando archivo local:", filepath)

# ================================================================
# 2. LEER SEÑAL DE LIGO
# ================================================================
def load_ligo(filepath, gps_ini, gps_fin):

    with h5py.File(filepath, "r") as f:
        strain = f["strain"]["Strain"][:]
        gps_start = f["meta"]["GPSstart"][()]
        duration = f["meta"]["Duration"][()]

    fs = float(len(strain) / duration)

    i0 = int(round((gps_ini - gps_start) * fs))
    i1 = int(round((gps_fin - gps_start) * fs))

    if i0 < 0 or i1 > len(strain):
        raise ValueError("El segmento pedido no está en el archivo.")

    x = np.asarray(strain[i0:i1], dtype=float)

    print(f"fs = {fs} Hz | N evento = {len(x)} | duración = {len(x) / fs:.2f} s")

    return x, fs

# ================================================================
# 3. PREPARAR LA SEÑAL
# ================================================================

def ajustar_senal(x, level):

    N_original = len(x)

    block = 2 ** level

    # Queremos una longitud divisible por 2^LEVEL.
    N = (N_original // block) * block

    if N < 4096:
        raise ValueError(
            f"La señal no posee suficientes muestras para el nivel de compresión requerido."
        )

    x = x[:N]

    # Eliminar valor medio.
    x = x - np.mean(x)

    return x

# ================================================================
# 4. FFT
# ================================================================

def compresion_fft(x, energy):

    N = len(x)

    # FFT completa
    X = np.fft.fft(x)

    # Energía de cada coeficiente
    coef_e = np.abs(X) ** 2

    coef_e[1:-1] *= 2 #Dado la forma de la fft

    # Ordenamos de mayor a menor energía
    ordenar = np.argsort(coef_e)[::-1]

    energia_cum= (
        np.cumsum(coef_e[ordenar])
        / np.sum(coef_e)
    )

    # Número mínimo de coeficientes necesarios
    k = np.searchsorted(energia_cum, energy) + 1

    # Conservamos solo los k más grandes
    k_conservar = ordenar[:k]

    X_comp = np.zeros_like(X)
    X_comp[k_conservar] = X[k_conservar]

    # Reconstrucción (irfft es la inversa de rfft)
    x_recons= np.fft.irfft(X_comp, n=N)

    number_coefficients = 2 * k

    return (
        X,
        X_comp,
        x_recons,
        number_coefficients,
        energia_cum,
        k_conservar
    )


# ================================================================
# 5. DWT
# ================================================================

def compresion_dwt(x, wavelet, level, energy):

    N = len(x)

    # DWT multirresolución
    coeficientes = pywt.wavedec(
        x,
        wavelet,
        level=level,
        mode="periodization"
    )

    # Convertimos todos los coeficientes a un único vector
    arr, slices = pywt.coeffs_to_array(coeficientes)

    # Energía de cada coeficiente
    coef_e = arr ** 2

    energia_total = np.sum(coef_e)

    # Orden de mayor a menor energía
    orden_energia= np.argsort(coef_e)[::-1]

    energia_cum = (
        np.cumsum(coef_e[orden_energia])
        / energia_total
    )

    # Número mínimo de coeficientes
    k = np.searchsorted(
        energia_cum,
        energy
    ) + 1

    # Conservamos solamente los k mayores
    arr_comp = np.zeros_like(arr)

    seleccion_k = orden_energia[:k]

    arr_comp[seleccion_k] = arr[seleccion_k]

    # Reconstruimos estructura de coeficientes
    coeffs_comp = pywt.array_to_coeffs(
        arr_comp,
        slices,
        output_format="wavedec"
    )

    # DWT inversa
    x_recons= pywt.waverec(
        coeffs_comp,
        wavelet,
        mode="periodization"
    )[:N]

    return (
        coeficientes,
        arr,
        arr_comp,
        x_recons,
        k,
        energia_cum,
        seleccion_k
    )


# ================================================================
# 6. ERROR CUADRÁTICO MEDIO
# ================================================================

def err_cuadratico_medio(x, xr):

    return np.mean(
        (x - xr) ** 2
    )

# ================================================================
# 7. K NECESARIOS
# ================================================================

def grafica_k_necesarios(cum_fft, cum_dwt, K_fft, K_dwt, energy, N):

    # Eje x de la FFT: cada coeficiente complejo cuenta como 2 reales
    n_fft = 2 * np.arange(1, len(cum_fft) + 1)

    # Eje x de la DWT: un coeficiente = un número real
    n_dwt = np.arange(1, len(cum_dwt) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(n_fft, cum_fft, "r", label=f"FFT: K = {K_fft}  (CR = {N / K_fft:.1f})")
    plt.plot(n_dwt, cum_dwt, "b", label=f"DWT {WAVELET}, L={LEVEL}: K = {K_dwt}  (CR = {N / K_dwt:.1f})")

    # Línea del umbral de energía
    plt.axhline(energy, color="gray", ls="--", label=f"{energy:.0%} de la energía")

    # Marcar el K de cada método
    plt.axvline(K_fft, color="r", ls=":")
    plt.axvline(K_dwt, color="b", ls=":")
    plt.plot(K_fft, energy, "ro")
    plt.plot(K_dwt, energy, "bo")

    plt.xscale("log")      # K es mucho menor que N, en log se aprecia mejor
    plt.xlabel("Número de coeficientes conservados (K)")
    plt.ylabel("Energía acumulada")
    plt.title("Coeficientes necesarios para alcanzar el umbral de energía")
    plt.legend()

    plt.tight_layout()
    plt.savefig("ej1_k_necesarios_GW190412.png", dpi=200)
# ================================================================
# 8. PROGRAMA PRINCIPAL
# ================================================================

def main():

    #en caso de tener archivo descargado, sino intenta descargarlo
    archivos_locales = glob.glob("H-H1_GWOSC_4KHZ_R1-1239082247-32.hdf5")

    if archivos_locales:
        filepath = archivos_locales[0]
        print("Usando archivo local:", filepath)

    x, fs = load_ligo(filepath, GPS_START, GPS_END)
    x = ajustar_senal(x, LEVEL)

    N = len(x)
    t = GPS_START + np.arange(N) / fs

    # FFT y DWT
    (X, X_comp, x_fft, K_fft, cum_fft, keep_fft) = compresion_fft(x, ENERGY)
    (coeffs, arr, arr_comp, x_dwt, K_dwt, cum_dwt, keep_dwt) = compresion_dwt(x, WAVELET, LEVEL, ENERGY)

    # MSE y CR
    mse_fft = err_cuadratico_medio(x, x_fft)
    mse_dwt = err_cuadratico_medio(x, x_dwt)

    print(f"\n{'Método':<10}{'Coef.':>10}{'CR = N/K':>12}{'MSE':>14}")
    print(f"{'FFT':<10}{K_fft:>10}{N / K_fft:>12.2f}{mse_fft:>14.3e}")
    print(f"{'DWT':<10}{K_dwt:>10}{N / K_dwt:>12.2f}{mse_dwt:>14.3e}")

    # Gráficas: original, reconstrucciones, residuos
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax[0].plot(t, x, "k")
    ax[0].set_title("Señal original de LIGO — evento GW190412")
    ax[0].set_ylabel("Strain")
    ax[0].grid(alpha=0.3)

    ax[1].plot(t, x, "k", alpha=0.5, label="Original")
    ax[1].plot(t, x_fft, "r", label=f"FFT ({K_fft} coef.)")
    ax[1].plot(t, x_dwt, "b", label=f"DWT {WAVELET}, L={LEVEL} ({K_dwt} coef.)")
    ax[1].set_title(f"Reconstrucciones con {ENERGY:.0%} de la energía")
    ax[1].set_ylabel("Strain")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    ax[2].plot(t, x - x_fft, "r", label=f"Residuo FFT, MSE={mse_fft:.2e}")
    ax[2].plot(t, x - x_dwt, "b", label=f"Residuo DWT, MSE={mse_dwt:.2e}")
    ax[2].set_title("Residuos: e[n] = x[n] - x_rec[n]")
    ax[2].set_xlabel("Tiempo [s] (GPS)")
    ax[2].set_ylabel("Error")
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("ej1_reconstruccion_evento_GW190412.png", dpi=200)
    
    grafica_k_necesarios(cum_fft, cum_dwt, K_fft, K_dwt, ENERGY, N)

    plt.show()
 
 

if __name__ == "__main__":
    main()
