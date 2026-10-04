import numpy as np
import matplotlib
matplotlib.use('Agg') # Agrega esta línea para que funcione en Codespaces
import matplotlib.pyplot as plt
import pywt
from skimage import io, color
from skimage.transform import resize


# PARTE 1: Adquisición e Inspección
nombre_archivo = 'webb.png' 

img_original = io.imread(nombre_archivo)

if img_original.ndim == 3:
    # Si tiene 4 canales (RGBA, típico en PNG), nos quedamos solo con los 3 primeros (RGB)
    if img_original.shape[-1] == 4:
        img_original = img_original[:, :, :3]
        
    img = color.rgb2gray(img_original)
else:
    img = img_original

if img.shape != (512, 512):
    img = resize(img, (512, 512), anti_aliasing=True)

# Inspección visual en escala logarítmica
plt.figure(figsize=(6, 5))
plt.imshow(np.log1p(img), cmap='magma')
plt.title("Parte 1: JWST Original (Escala Logarítmica)")
plt.colorbar()
plt.savefig('grafico_parte1_original.png', bbox_inches='tight')
plt.close()
print("Gráfico de la Parte 1 guardado.")

# PARTE 2: 2D-DWT (Transformada Wavelet)
wavelet_type = 'db4' # Puede ser 'bior2.2', 'db2' o 'db4'
L = 2

# Descomposición Wavelet a nivel 2
coeffs = pywt.wavedec2(img, wavelet_type, level=L)
cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

# Visualización de subbandas (Aproximación y Detalles L=2)
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0, 0].imshow(cA2, cmap='gray')
axs[0, 0].set_title('Aproximación cA2')
axs[0, 1].imshow(cH2, cmap='gray')
axs[0, 1].set_title('Detalle Horiz. cH2')
axs[1, 0].imshow(cV2, cmap='gray')
axs[1, 0].set_title('Detalle Vert. cV2')
axs[1, 1].imshow(cD2, cmap='gray')
axs[1, 1].set_title('Detalle Diag. cD2')
plt.suptitle("Parte 2: Subbandas 2D-DWT (Nivel 2)")
plt.tight_layout()
plt.savefig('grafico_parte2_subbandas.png', bbox_inches='tight')
plt.close()
print("Gráfico de la Parte 2 guardado.")


# PARTE 3: 2D-FFT (Transformada de Fourier)

F = np.fft.fft2(img)
F_shift = np.fft.fftshift(F) # Centrar el espectro

mag_spectrum = np.log1p(np.abs(F_shift))
phase_spectrum = np.angle(F_shift)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(mag_spectrum, cmap='magma')
axs[0].set_title('Espectro de Magnitud log(1 + |F(u,v)|)')
axs[1].imshow(phase_spectrum, cmap='inferno')
axs[1].set_title('Espectro de Fase arg(F(u,v))')
plt.suptitle("Parte 3: Análisis 2D-FFT")
plt.savefig('grafico_parte3_espectros.png', bbox_inches='tight')
plt.close()
print("Gráfico de la Parte 3 guardado.")

# PARTE 4: Aislamiento con Wavelets

# 1. Anular la aproximación de baja frecuencia
cA2_mod = np.zeros_like(cA2)

# 2. Función de umbralizado suave (Soft Thresholding)
def soft_threshold(w, lam):
    return np.sign(w) * np.maximum(np.abs(w) - lam, 0)

# Aplicar umbralizado a los coeficientes de detalle
# El valor lambda (lam) depende de la intensidad de la imagen
lam = 0.5 
details_mod = []
for level_details in coeffs[1:]:
    cH, cV, cD = level_details
    details_mod.append((soft_threshold(cH, lam),
                        soft_threshold(cV, lam),
                        soft_threshold(cD, lam)))

# 3. Reconstrucción mediante 2D-IDWT
coeffs_mod = [cA2_mod] + details_mod
f_wavelet = pywt.waverec2(coeffs_mod, wavelet_type)

# PARTE 5: Aislamiento con 2D-FFT
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
u = np.arange(rows) - crow
v = np.arange(cols) - ccol
U, V = np.meshgrid(v, u)

# 1. Diseñar Filtro Pasa-Altas Gaussiano
D2 = U*2 + V*2
D0 = 15.0 # Radio de corte (ajustable)
H = 1 - np.exp(-D2 / (2 * D0**2))

# 2. Filtrado y Transformada Inversa
F_filtered_shift = F_shift * H
F_filtered = np.fft.ifftshift(F_filtered_shift)
f_fft = np.fft.ifft2(F_filtered).real

# PARTE 6: Evaluación Comparativa

# Construir panel 2x2
diferencia_absoluta = np.abs(f_wavelet - f_fft)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Se usa un vmin/vmax para hacer comparable la visualización
axs[0, 0].imshow(img, cmap='magma')
axs[0, 0].set_title('Original')

axs[0, 1].imshow(f_wavelet, cmap='magma', vmin=0)
axs[0, 1].set_title('$f_{wavelet}$ (Soft Threshold + No cA2)')

axs[1, 0].imshow(f_fft, cmap='magma', vmin=0)
axs[1, 0].set_title('$f_{fft}$ (Filtro Gaussiano Pasa-Altas)')

axs[1, 1].imshow(diferencia_absoluta, cmap='inferno')
axs[1, 1].set_title('Diferencia Absoluta $|f_{wavelet} - f_{fft}|$')

for ax in axs.flat:
    ax.axis('off')

plt.suptitle("Parte 6: Comparación de Aislamiento de Fuentes Puntuales", fontsize=16)
plt.tight_layout()
plt.savefig('grafico_parte6_comparacion.png', bbox_inches='tight')
plt.close()
print("Gráfico de la Parte 6 guardado. ¡Proceso terminado!")
