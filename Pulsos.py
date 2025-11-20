import cv2
import numpy as np
from scipy.signal import find_peaks

# === CONFIGURACIONES ===
VIDEO_PATH = r'E:\Reconocimiento\microscopicos\Control A\Control A_3.mp4'  #Windows
#VIDEO_PATH = r'/run/media/lurtz/Automethic/Reconocimiento/Microscopicos/Control A/Control A.mp4'  #linux

# ===== Control A
#VIDEO_PATH = r'E:\Reconocimiento\microscopicos\Control A\Control A.mp4'    #158-164 BPM
#VIDEO_PATH = r'E:\Reconocimiento\microscopicos\Control A\Control A_1.mp4'  #150-152 BPM
#VIDEO_PATH = r'E:\Reconocimiento\microscopicos\Control A\Control A_2.mp4'  #127-135 BPM
#VIDEO_PATH = r'E:\Reconocimiento\microscopicos\Control A\Control A_3.mp4'  #113-118 BPM
#VIDEO_PATH = r'/run/media/lurtz/Automethic/Reconocimiento/Microscopicos/Control A/Control A_3.mp4'  #linux

#VIDEO_PATH = r'E:\Reconocimiento\microscopicos\Control A\Control A_4.mp4'   #131-131 BPM

ACCELERATED = True                # True = análisis rápido (sin mostrar todos los frames)
ROI_SELECTED = False

# === ABRIR VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or np.isnan(fps):
    fps = 30  # valor por defecto
print(f"FPS detectado: {fps}")

# === SELECCIONAR ROI (Zona del corazón) ===
ret, frame = cap.read()
if not ret:
    print("No se pudo leer el video.")
    cap.release()
    exit()

roi_box = cv2.selectROI("Selecciona la zona del corazon", frame, fromCenter=False, showCrosshair=False)
cv2.destroyWindow("Selecciona la zona del corazon")

x, y, w, h = map(int, roi_box)
intensity_values = []

# === PROCESAMIENTO DE FRAMES ===
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Promedio de intensidad de brillo del ROI
    mean_intensity = np.mean(gray)
    intensity_values.append(mean_intensity)

    # Mostrar en tiempo real si no está acelerado
    if not ACCELERATED:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# === EVITAR DIVISIONES POR CERO ===
if len(intensity_values) == 0:
    print("No se obtuvieron valores de intensidad.")
    exit()

# === PROCESAMIENTO DE SEÑAL ===
signal = np.array(intensity_values)
signal = signal - np.mean(signal)
signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

# Detectar picos (pulsaciones)
peaks, _ = find_peaks(signal, distance=fps/4, prominence=0.01)

# === CALCULO DE FRECUENCIA CARDIACA ===
duration_sec = len(signal) / fps
if duration_sec == 0:
    print("Duración del video = 0 segundos.")
    exit()

heart_rate = len(peaks) * (60 / duration_sec)
print(f"Frecuencia cardíaca estimada: {heart_rate:.1f} BPM")

with open("BMPS.csv",'a') as f:
    f.write(VIDEO_PATH+','+str(heart_rate)+'\n')

# === MOSTRAR RESULTADOS ===
import matplotlib.pyplot as plt

time_axis = np.linspace(0, duration_sec, len(signal))

plt.figure(figsize=(10, 5))
plt.plot(time_axis, signal, label="Señal normalizada (intensidad ROI)")
plt.plot(time_axis[peaks], signal[peaks], "rx", label="Pulsaciones detectadas")
plt.title(f"Frecuencia cardiaca estimada: {heart_rate:.1f} BPM")
plt.xlabel("Tiempo (s)")
plt.ylabel("Intensidad normalizada")
plt.legend()
plt.grid()
plt.show()
