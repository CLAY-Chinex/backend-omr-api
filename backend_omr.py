import cv2
import numpy as np

# =============================================================================
#  CLASE DE PROCESAMIENTO OMR (CEREBRO)
# =============================================================================

class OMRProcessor:
    def ordenar_puntos(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def transformar_perspectiva(self, imagen, pts):
        rect = self.ordenar_puntos(pts)
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(imagen, M, (maxWidth, maxHeight))
        return warped

    def detectar_hoja(self, img_original):
        gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
        
        return None

    def procesar_imagen(self, ruta_imagen):
        try:
            # Lectura robusta
            stream = open(ruta_imagen, "rb")
            bytes_data = bytearray(stream.read())
            numpyarray = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            stream.close()
        except:
            return None, "Error al leer archivo"

        if img is None: return None, "Imagen corrupta"

        # 1. Detección de Hoja (Contorno Grande)
        puntos_hoja = self.detectar_hoja(img)
        if puntos_hoja is None:
            return None, "No se encontró el borde de la hoja"

        # 2. Transformación de Perspectiva
        img_warp = self.transformar_perspectiva(img, puntos_hoja)
        img_warp_gray = cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY)
        
        # Binarización (Otsu)
        thresh = cv2.threshold(img_warp_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        return thresh, "OK"

    def leer_bloque_respuestas(self, img_thresh, preguntas=30, opciones=5):
        # Asumimos que img_thresh es SOLO la columna de burbujas recortada
        (h, w) = img_thresh.shape
        step_y = h / preguntas
        step_x = w / opciones
        respuestas = []

        for i in range(preguntas):
            start_y = int(i * step_y)
            end_y = int((i + 1) * step_y)
            
            fila_counts = []
            for j in range(opciones):
                start_x = int(j * step_x)
                end_x = int((j + 1) * step_x)
                
                mask = img_thresh[start_y:end_y, start_x:end_x]
                fila_counts.append(cv2.countNonZero(mask))

            # Umbral dinámico: si el pixelado es mayor a cierto valor
            # Aquí simplificado: tomamos el índice con más pixeles blancos
            max_val = max(fila_counts)
            if max_val > 100: # Mínimo de tinta para considerar marca
                idx = fila_counts.index(max_val)
                respuestas.append(idx)
            else:
                respuestas.append(-1) # Vacío
        
        return respuestas

    def analizar_examen_completo(self, img_bin):
        # --- DEFINICIÓN DE ROIs (Regiones de Interés) ---
        # Estos valores son porcentajes relativos al tamaño de la hoja A4 escaneada
        h, w = img_bin.shape
        
        # Coordenadas aproximadas (Ajustar según tu diseño de hoja real)
        # Código Estudiante (Parte Superior Izquierda)
        roi_cod_x = int(w * 0.09)
        roi_cod_y = int(h * 0.30)
        roi_cod_w = int(w * 0.25)
        roi_cod_h = int(h * 0.30)
        
        # Columna 1 (Preguntas 1-30)
        roi_col1_x = int(w * 0.45)
        roi_col1_y = int(h * 0.065)
        roi_col1_w = int(w * 0.19)
        roi_col1_h = int(h * 0.89)

        # Columna 2 (Preguntas 31-60)
        roi_col2_x = int(w * 0.71)
        roi_col2_y = int(h * 0.065)
        roi_col2_w = int(w * 0.19)
        roi_col2_h = int(h * 0.89)

        # Recortes
        crop_codigo = img_bin[roi_cod_y:roi_cod_y+roi_cod_h, roi_cod_x:roi_cod_x+roi_cod_w]
        crop_col1 = img_bin[roi_col1_y:roi_col1_y+roi_col1_h, roi_col1_x:roi_col1_x+roi_col1_w]
        crop_col2 = img_bin[roi_col2_y:roi_col2_y+roi_col2_h, roi_col2_x:roi_col2_x+roi_col2_w]

        # Lectura
        res_col1 = self.leer_bloque_respuestas(crop_col1, preguntas=30)
        res_col2 = self.leer_bloque_respuestas(crop_col2, preguntas=30)
        
        # Combinar resultados (Total 60)
        respuestas_raw = res_col1 + res_col2
        
        # Convertir índices a Letras
        letras = ["A", "B", "C", "D", "E"]
        respuestas_finales = []
        for r in respuestas_raw:
            if r == -1: respuestas_finales.append("")
            else: respuestas_finales.append(letras[r])
            
        # Simulación de lectura de código (Para simplificar este ejemplo server)
        # En una versión real, aquí aplicarías lógica similar a las preguntas
        codigo_detectado = "123456" 

        return codigo_detectado, respuestas_finales

# Instancia global para usar en main.py
processor = OMRProcessor()