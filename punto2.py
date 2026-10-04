import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage import io, color
from skimage.transform import resize

# ==========================================
# PARTE 1: Adquisición e Inspección
# ==========================================
# OPCIÓN A: Cargar tu propia imagen (Descomenta las siguientes 3 líneas y pon tu ruta)
# ruta_imagen = 'jwst_image.png' 
# img = color.rgb2gray(io.imread(ruta_imagen))
# img = resize(img, (512, 512)) # Asegurar 512x512

# OPCIÓN B: Simular una imagen tipo JWST (Campo profundo con fuentes puntuales)
np.random.seed(42)
img = np.zeros((512, 512))
# Agregar ruido de fondo galáctico de baja frecuencia
X, Y = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512))
fondo = 0.5 * np.exp(-(X*2 + Y*2) / 0.5)
# Agregar fuentes puntuales (estrellas/galaxias)
for _ in range(100):
    x, y = np.random.randint(0, 512, 2)
    img[x, y] = np.random.uniform(5, 25)
from scipy.ndimage import gaussian_filter
img = fondo + gaussian_filter(img, sigma=1.2) # Aplicar PSF

# Inspección visual en escala logarítmica
plt.figure(figsize=(6, 5))
plt.imshow(np.log1p(img), cmap='magma')
plt.title("Parte 1: JWST Original (Escala Logarítmica)")
plt.colorbar()
plt.show()

# ==========================================
# PARTE 2: 2D-DWT (Transformada Wavelet)
# ==========================================
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
plt.show()

# ==========================================
# PARTE 3: 2D-FFT (Transformada de Fourier)
# ==========================================
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
plt.show()

# ==========================================
# PARTE 4: Aislamiento con Wavelets
# ==========================================
# 1. Anular la aproximación de baja frecuencia
cA2_mod = np.zeros_like(cA2)

# 2. Función de umbralizado suave (Soft Thresholding)
def soft_threshold(w, lam):
    return np.sign(w) * np.maximum(np.abs(w) - lam, 0)

# Aplicar umbralizado a los coeficientes de detalle
# El valor lambda (lam) depende de la intensidad de tu imagen; ajustarlo si es necesario
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

# ==========================================
# PARTE 5: Aislamiento con 2D-FFT
# ==========================================
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

# ==========================================
# PARTE 6: Evaluación Comparativa
# ==========================================
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
plt.show()
