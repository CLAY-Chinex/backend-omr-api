import cv2
import numpy as np
import logging

# =============================================================================
# CONFIGURACIÓN DEL SISTEMA
# =============================================================================

# Configuración de logs para ver errores en la consola de Render
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OMRConfig:
    """
    Parámetros de Calibración del OMR.
    Estos valores han sido ajustados en el prototipo para mejorar la detección.
    """
    # --- Visión Artificial ---
    UMBRAL_BINARIZACION = 61      # BlockSize para Adaptive Threshold
    CONSTANTE_C = 15              # Constante para Adaptive Threshold
    AREA_MIN_ANCLA = 800          # Área mínima para considerar un cuadrado (px)
    FACTOR_RELLENO_MIN = 0.40     # Solidez del cuadrado (evita falsos positivos)
    
    # --- Lógica de Respuesta ---
    UMBRAL_PRESENCIA = 160        # Píxeles blancos mínimos para considerar marcado
    
    # --- Geometría de la Hoja (Normalizada 0.0 a 1.0) ---
    ROI_CODIGO = {'x_in': 0.091, 'x_fn': 0.338, 'y_in': 0.303, 'y_fn': 0.602}
    ROI_COL1   = {'x_in': 0.4509, 'x_fn': 0.6434, 'y_in': 0.065, 'y_fn': 0.958}
    ROI_COL2   = {'x_in': 0.7135, 'x_fn': 0.906,  'y_in': 0.065, 'y_fn': 0.958}

    # --- Estructura del Examen ---
    PREGUNTAS_POR_COLUMNA = 30
    OPCIONES_POR_PREGUNTA = 5
    DIGITOS_CODIGO = 8
    FILAS_CODIGO = 10


class OMRProcessor:
    """
    Motor de procesamiento de imágenes.
    Encapsula toda la lógica de visión artificial.
    """
    def __init__(self):
        self.cfg = OMRConfig()

    # -------------------------------------------------------------------------
    # UTILIDADES INTERNAS
    # -------------------------------------------------------------------------
    def _ordenar_puntos(self, pts):
        """Ordena coordenadas en orden: Top-Left, Top-Right, Bottom-Right, Bottom-Left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _cargar_imagen(self, ruta):
        """Lectura segura de imágenes (soporta caracteres especiales y rutas complejas)"""
        try:
            with open(ruta, "rb") as stream:
                bytes_data = bytearray(stream.read())
                numpyarray = np.asarray(bytes_data, dtype=np.uint8)
                img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                return img
        except Exception as e:
            logger.error(f"Error cargando imagen {ruta}: {e}")
            return None

    # -------------------------------------------------------------------------
    # ETAPA 1: DETECCIÓN Y TRANSFORMACIÓN
    # -------------------------------------------------------------------------
    def procesar_hoja_completa(self, ruta_imagen):
        """
        Detecta la hoja, corrige perspectiva y rotación.
        Retorna: (bool_exito, img_binaria_final, mensaje_error)
        Nota: Ya no retornamos img_original para ahorrar memoria en el server.
        """
        img = self._cargar_imagen(ruta_imagen)
        if img is None:
            return False, None, "ERROR_LECTURA_ARCHIVO"

        h_img, w_img = img.shape[:2]
        area_total = h_img * w_img

        # 1. Preprocesamiento
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Binarización Adaptativa (Clave para corregir sombras)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, self.cfg.UMBRAL_BINARIZACION, self.cfg.CONSTANTE_C
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 2. Buscar Contornos (Anclas)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidatos = []

        for c in cnts:
            area = cv2.contourArea(c)
            # Descartar ruido pequeño o manchas gigantes
            if area < self.cfg.AREA_MIN_ANCLA or area > (area_total * 0.05):
                continue
            
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True)

            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                # Verificar que sea cuadrado
                if 0.8 <= aspect_ratio <= 1.2:
                    # Verificar Solidez (relleno de tinta)
                    roi = thresh[y:y+h, x:x+w]
                    relleno = cv2.countNonZero(roi) / (w * h)
                    
                    if relleno > self.cfg.FACTOR_RELLENO_MIN:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            candidatos.append((area, [cX, cY], c, max(w, h)))

        # Ordenar por área y tomar los 4 más grandes
        candidatos.sort(key=lambda x: x[0], reverse=True)
        anclas = candidatos[:4]

        if len(anclas) != 4:
            logger.warning(f"Anclas detectadas: {len(anclas)}. Se requieren 4.")
            return False, None, "ERROR_ANCLAS"

        # 3. Transformación de Perspectiva
        pts = np.array([item[1] for item in anclas], dtype="float32")
        tam_promedio = np.mean([item[3] for item in anclas])

        img_aplanada_gris = self._transformar_perspectiva(gray, pts, tam_promedio)

        # 4. Autocorrección de Orientación
        img_aplanada_gris = self._corregir_orientacion(img_aplanada_gris)

        # 5. Generar Binaria Final para análisis
        # Usamos Otsu aquí porque la iluminación ya debería ser uniforme tras el warp
        _, img_binaria_final = cv2.threshold(
            img_aplanada_gris, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        return True, img_binaria_final, "OK"

    def _transformar_perspectiva(self, imagen, pts, tam_cuadro):
        rect = self._ordenar_puntos(pts)
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
        widthB = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))
        heightB = np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))
        maxHeight = max(int(heightA), int(heightB))

        # Margen dinámico basado en el tamaño de las anclas
        margen = int(tam_cuadro * 0.52)

        dst = np.array([
            [margen, margen],
            [maxWidth - 1 + margen, margen],
            [maxWidth - 1 + margen, maxHeight - 1 + margen],
            [margen, maxHeight - 1 + margen]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(imagen, M, (maxWidth + 2*margen, maxHeight + 2*margen))

    def _corregir_orientacion(self, img):
        h, w = img.shape
        banda = 0.025 # 2.5% del borde para detectar tinta
        
        def densidad(roi):
            # Binarización rápida local para contar tinta
            _, b = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            return cv2.countNonZero(b)

        rois = {
            "sup": img[0:int(h*banda), :],
            "inf": img[int(h*(1-banda)):h, :],
            "izq": img[:, 0:int(w*banda)],
            "der": img[:, int(w*(1-banda)):w]
        }
        
        densidades = {k: densidad(v) for k, v in rois.items()}
        lado = max(densidades, key=densidades.get)
        
        if lado == "sup": return cv2.rotate(img, cv2.ROTATE_180)
        elif lado == "izq": return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif lado == "der": return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    # -------------------------------------------------------------------------
    # ETAPA 2: EXTRACCIÓN DE DATOS
    # -------------------------------------------------------------------------
    def analizar_examen(self, img_binaria):
        """
        Procesa la imagen ya enderezada y devuelve los datos crudos.
        """
        H, W = img_binaria.shape
        
        # Función auxiliar de recorte
        def get_roi(cfg_roi):
            return img_binaria[
                int(H*cfg_roi['y_in']):int(H*cfg_roi['y_fn']),
                int(W*cfg_roi['x_in']):int(W*cfg_roi['x_fn'])
            ]

        # 1. Recortar
        crop_cod = get_roi(self.cfg.ROI_CODIGO)
        crop_c1 = get_roi(self.cfg.ROI_COL1)
        crop_c2 = get_roi(self.cfg.ROI_COL2)

        # 2. Leer
        codigo_str = self._leer_matriz_codigo(crop_cod)
        respuestas_c1 = self._leer_bloque_preguntas(crop_c1)
        respuestas_c2 = self._leer_bloque_preguntas(crop_c2)
        
        # 3. Formatear respuestas
        opciones = ['A', 'B', 'C', 'D', 'E']
        lista_final = []
        
        for r in respuestas_c1 + respuestas_c2:
            if r == -1: lista_final.append("")        # Vacía
            elif r == -2: lista_final.append("ANULADA") # Error Múltiple
            else: lista_final.append(opciones[r])
            
        return codigo_str, lista_final

    def _leer_matriz_codigo(self, roi):
        H, W = roi.shape
        step_x = W / self.cfg.DIGITOS_CODIGO
        step_y = H / self.cfg.FILAS_CODIGO
        codigo = ""

        for j in range(self.cfg.DIGITOS_CODIGO):
            x_start = int(j * step_x); x_end = int((j + 1) * step_x)
            col_scores = []

            for i in range(self.cfg.FILAS_CODIGO):
                y_start = int(i * step_y); y_end = int((i + 1) * step_y)
                celda = roi[y_start:y_end, x_start:x_end]
                col_scores.append(cv2.countNonZero(celda))
            
            marcados = [idx for idx, val in enumerate(col_scores) if val > self.cfg.UMBRAL_PRESENCIA]

            if len(marcados) == 1:
                codigo += str(marcados[0])
            elif len(marcados) == 0:
                codigo += "?"
            else:
                return "INVALIDO" # Código inválido si hay múltiples marcas en una columna

        return codigo

    def _leer_bloque_preguntas(self, roi):
        H, W = roi.shape
        step_y = H / self.cfg.PREGUNTAS_POR_COLUMNA
        step_x = W / self.cfg.OPCIONES_POR_PREGUNTA
        respuestas = []

        for i in range(self.cfg.PREGUNTAS_POR_COLUMNA):
            y_start = int(i * step_y); y_end = int((i + 1) * step_y)
            row_scores = []

            for j in range(self.cfg.OPCIONES_POR_PREGUNTA):
                x_start = int(j * step_x); x_end = int((j + 1) * step_x)
                celda = roi[y_start:y_end, x_start:x_end]
                row_scores.append(cv2.countNonZero(celda))

            marcados = [idx for idx, val in enumerate(row_scores) if val > self.cfg.UMBRAL_PRESENCIA]

            if len(marcados) == 1:
                respuestas.append(marcados[0])
            elif len(marcados) == 0:
                respuestas.append(-1) # Vacío
            else:
                respuestas.append(-2) # Error Múltiple

        return respuestas


# ... (código de la clase OMRProcessor arriba) ...

# =============================================================================
# API PÚBLICA (WRAPPER DE COMPATIBILIDAD)
# =============================================================================

def procesar_imagen_examen(ruta_imagen):
    """
    Función puente que instancia la clase y procesa.
    """
    processor = OMRProcessor() # <--- Aquí se crea el 'processor' internamente
    
    # 1. Procesamiento de imagen
    exito, img_binaria, mensaje = processor.procesar_hoja_completa(ruta_imagen)
    
    if not exito:
        # Nota: Usamos logging en lugar de print
        logging.error(f"Fallo en procesamiento: {mensaje}") 
        return mensaje, [] 

    # 2. Lectura de datos
    try:
        codigo_str, lista_respuestas = processor.analizar_examen(img_binaria)
        return codigo_str, lista_respuestas
        
    except Exception as e:
        logging.error(f"Error en análisis: {e}")
        return "ERROR_PROCESAMIENTO", []