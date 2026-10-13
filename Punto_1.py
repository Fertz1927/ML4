
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pywt
import glob



# ================================================================
# CONFIGURACIÓN
# ================================================================
ENERGY = 0.95        # 95 % de la energía
WAVELET = "db4"        # familia ortogonal
LEVEL = 5              # nivel dwt

# ================================================================
# 1. Señal de LIGO
# ================================================================

EVENTOS = [
    {
        "nombre": "GW150914",
        "archivo": "H-H1_LOSC_4_V1-1126256640-4096.hdf5",
        "gps_start": 1126259460,
        "gps_end": 1126259464,
    },
    {
        "nombre": "GW190412",
        "archivo": "H-H1_GWOSC_4KHZ_R1-1239082247-32.hdf5",
        "gps_start": 1239082260,
        "gps_end": 1239082264,
    },
]

# ================================================================
# 2. LEER SEÑAL DE LIGO
# ================================================================

def load_ligo(filepath,gps_ini, gps_fin):

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
 
    # Reconstrucción (ifft es la inversa de fft)
    x_recons= np.fft.ifft(X_comp, n=N)
 
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

def grafica_k_necesarios(cum_fft, cum_dwt, K_fft, K_dwt, energy, N, nombre):

    n_fft = 2 * np.arange(1, len(cum_fft) + 1)
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
    plt.savefig(f"{nombre}_k_necesarios.png", dpi=200)

# ================================================================
# 8. ANALIZAR EVENTO
# ================================================================


def analizar_evento(evento):
    nombre = evento["nombre"]
    print(f"\n{'=' * 65}\nEvento: {nombre}\nArchivo: {evento['archivo']}")

    x, fs = load_ligo(
        evento["archivo"], evento["gps_start"], evento["gps_end"]
    )
    x = ajustar_senal(x, LEVEL)

    N = len(x)
    t = evento["gps_start"] + np.arange(N) / fs

    _, _, x_fft, K_fft, cum_fft, _ = compresion_fft(x, ENERGY)
    _, _, _, x_dwt, K_dwt, cum_dwt, _ = compresion_dwt(
        x, WAVELET, LEVEL, ENERGY
    )

    mse_fft = err_cuadratico_medio(x, x_fft)
    mse_dwt = err_cuadratico_medio(x, x_dwt)
    cr_fft = N / K_fft
    cr_dwt = N / K_dwt

    print(f"\n{'Método':<10}{'Coef. guardados':>18}{'CR = N/K':>12}{'MSE':>14}")
    print(f"{'FFT':<10}{K_fft:>18}{cr_fft:>12.2f}{mse_fft:>14.3e}")
    print(f"{'DWT':<10}{K_dwt:>18}{cr_dwt:>12.2f}{mse_dwt:>14.3e}")

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax[0].plot(t, x, color="black")
    ax[0].set_title(f"Señal original de LIGO — {nombre}")
    ax[0].set_ylabel("Strain")
    ax[0].grid(alpha=0.3)

    ax[1].plot(t, x, color="black", alpha=0.45, label="Original")
    ax[1].plot(t, x_fft, label=f"FFT ({K_fft} números reales)")
    ax[1].plot(
        t, x_dwt, label=f"DWT {WAVELET}, L={LEVEL} ({K_dwt} coef.)"
    )
    ax[1].set_title(f"Reconstrucciones con {ENERGY:.0%} de energía")
    ax[1].set_ylabel("Strain")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    ax[2].plot(t, x - x_fft, label=f"Residuo FFT, MSE={mse_fft:.2e}")
    ax[2].plot(t, x - x_dwt, label=f"Residuo DWT, MSE={mse_dwt:.2e}")
    ax[2].set_title("Residuos: e[n] = x[n] - x_rec[n]")
    ax[2].set_xlabel("Tiempo GPS [s]")
    ax[2].set_ylabel("Error")
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{nombre}_reconstruccion_evento.png", dpi=200)

    grafica_k_necesarios(
        cum_fft, cum_dwt, K_fft, K_dwt, ENERGY, N, nombre
    )


# ================================================================
# 9. EJECUTAR LOS DOS EVENTOS
# ================================================================

def main():
    for evento in EVENTOS:
        try:
            analizar_evento(evento)
        except (FileNotFoundError, ValueError, KeyError, OSError) as error:
            print(f"\nNo se pudo analizar {evento['nombre']}: {error}")

    plt.show()
 
 
if __name__ == "__main__":
    main()
