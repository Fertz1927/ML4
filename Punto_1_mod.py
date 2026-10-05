import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pywt
import glob
import requests_pelican as rp
from gwdatafind import find_urls


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

def load_ligo(filepath, gps_start_0, gps_end_1):

    with h5py.File(filepath, "r") as f:

        # Estructura típica de los archivos LOSC/GWOSC
        strain = f["strain"]["Strain"][:]

        # Información temporal
        gps_start = f["meta"]["GPSstart"][()]
        duration = f["meta"]["Duration"][()]

    fs = float(len(strain) / duration)

    #Segmento de tiempo deseado

    i0 = int(round((gps_start_0 - gps_start) * fs))
    i1 = int(round((gps_end_1 - gps_start) * fs))
    if i0 < 0 or i1 > len(strain):
        raise ValueError("No se encuentra el segmento solicitado.")

    strain = strain[i0:i1]

    print("\nInformación de la señal")
    print("-----------------------")
    print(f"GPS inicial : {gps_start}")
    print(f"GPS deseado : {gps_end_1} - {gps_start_0}")
    print(f"Frecuencia  : {fs} Hz")
    print(f"N original  : {len(strain)}")
    print(f"Duración    : {len(strain) / fs:.2f} s")

    return np.asarray(strain, dtype=float), fs


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
# 7. PROGRAMA PRINCIPAL
# ================================================================

def main():
 
    #en caso de tener archivo dscargado, sino intenta descargarlo
    archivos_locales = glob.glob("H-H1_GWOSC_4KHZ_R1-1239082247-32.hdf5")
 
    if archivos_locales:
        filepath = archivos_locales[0]
        print("Usando archivo local:", filepath)

 
    x, fs = load_ligo(filepath, GPS_START, GPS_END)
    x = ajustar_senal(x, LEVEL)
 
    N = len(x)
    t = np.arange(N) / fs        # t = 0 corresponde a GPS_START
 
    print(f"\nSeñal utilizada: N = {N}, fs = {fs} Hz, duración = {N / fs:.2f} s")

 
    # ------------------------------------------------------------
    # Compresión con FFT y con DWT
    # ------------------------------------------------------------
 
    (X, X_comp, x_fft,
     K_fft, cum_fft, keep_fft) = compresion_fft(x, ENERGY)
 
    (coeffs, arr, arr_comp, x_dwt,
     K_dwt, cum_dwt, keep_dwt) = compresion_dwt(x, WAVELET, LEVEL, ENERGY)
 
    # ------------------------------------------------------------
    # Error cuadrático medio y ratio de compresión
    # ------------------------------------------------------------
 
    mse_fft = err_cuadratico_medio(x, x_fft)
    mse_dwt = err_cuadratico_medio(x, x_dwt)
 
    CR_fft = N / K_fft
    CR_dwt = N / K_dwt
 
    print("\n" + "=" * 60)
    print(f"{'Método':<10}{'Coeficientes':>15}{'CR = N/K':>12}{'MSE':>18}")
    print("-" * 60)
    print(f"{'FFT':<10}{K_fft:>15}{CR_fft:>12.2f}{mse_fft:>18.3e}")
    print(f"{'DWT':<10}{K_dwt:>15}{CR_dwt:>12.2f}{mse_dwt:>18.3e}")
    print("=" * 60)
 
    # ------------------------------------------------------------
    # Figura 1: original, reconstrucciones y residuos
    # ------------------------------------------------------------
 
    t_evento = GPS_EVENT - GPS_START          # instante de la fusión en nuestra ventana
    zoom = np.abs(t - t_evento) < 0.2         # +-0.2 s alrededor de la fusión
 
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
 
    ax[0].plot(t, x, "k")
    ax[0].set_title("Señal original de LIGO (H1)")
    ax[0].set_ylabel("Strain")
    ax[0].grid(alpha=0.3)
 
    ax[1].plot(t[zoom], x[zoom], "k", alpha=0.5, label="Original")
    ax[1].plot(t[zoom], x_fft[zoom], "r", label=f"FFT ({K_fft} coef.)")
    ax[1].plot(t[zoom], x_dwt[zoom], "b", label=f"DWT {WAVELET}, L={LEVEL} ({K_dwt} coef.)")
    ax[1].set_title(f"Reconstrucciones con {ENERGY:.0%} de la energía (zoom en la fusión)")
    ax[1].set_ylabel("Strain")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
 
    ax[2].plot(t[zoom], (x - x_fft)[zoom], "r", label=f"Residuo FFT, MSE={mse_fft:.2e}")
    ax[2].plot(t[zoom], (x - x_dwt)[zoom], "b", label=f"Residuo DWT, MSE={mse_dwt:.2e}")
    ax[2].set_title("Residuos: e[n] = x[n] - x_rec[n]")
    ax[2].set_xlabel("Tiempo [s]")
    ax[2].set_ylabel("Error")
    ax[2].legend()
    ax[2].grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(("ej1_reconstruccion_1.png"), dpi=200)
 
    # ------------------------------------------------------------
    # Figura 2: compactación de energía
    # ------------------------------------------------------------
 
    plt.figure(figsize=(9, 5))
 
    plt.semilogx(np.arange(1, len(cum_fft) + 1), cum_fft, label="FFT")
    plt.semilogx(np.arange(1, len(cum_dwt) + 1), cum_dwt, label=f"DWT ({WAVELET})")
    plt.axhline(ENERGY, linestyle="--", color="gray", label="95 %")
 
    plt.xlabel("Número de coeficientes (ordenados por energía)")
    plt.ylabel("Fracción acumulada de energía")
    plt.title("Compactación de energía")
    plt.legend()
    plt.grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(("ej1_compactacion_energia_1.png"), dpi=200)
 
    # ------------------------------------------------------------
    # Guardar los vectores comprimidos (para el Integrante 2)
    # ------------------------------------------------------------
 
    np.savez(
        os.path.join("vectores_comprimidos_1.npz"),
        x=x, fs=fs, N=N,
        X_fft=X_comp,          # FFT comprimida (formato rfft)
        coef_dwt=arr_comp      # coeficientes DWT comprimidos (un solo vector)
    )
 
    plt.show()
 
 
if __name__ == "__main__":
    main()
