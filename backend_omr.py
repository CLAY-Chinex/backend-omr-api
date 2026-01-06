import cv2
import numpy as np

# =============================================================================
#    UTILIDADES GEOMÉTRICAS
# =============================================================================

def ordenar_puntos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def transformar_perspectiva_dinamica(imagen, puntos_esquinas, tamaño_cuadro_ref):
    rect = ordenar_puntos(puntos_esquinas)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    margen_auto = int(tamaño_cuadro_ref * 0.52)
    dst = np.array([
        [margen_auto, margen_auto],
        [maxWidth - 1 + margen_auto, margen_auto],
        [maxWidth - 1 + margen_auto, maxHeight - 1 + margen_auto],
        [margen_auto, maxHeight - 1 + margen_auto]], dtype="float32")
    ancho_final = maxWidth + (margen_auto * 2)
    alto_final = maxHeight + (margen_auto * 2)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(imagen, M, (ancho_final, alto_final))
    return warped

# =============================================================================
#    VISIÓN ARTIFICIAL Y DETECCIÓN
# =============================================================================

def detectar_anchas_y_transformar(ruta_imagen):
    # Lectura robusta
    try:
        stream = open(ruta_imagen, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img_original = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        stream.close()
    except:
        return None, None

    if img_original is None: return None, None
    
    img_procesar = img_original.copy()
    gray = cv2.cvtColor(img_procesar, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    todos_los_cuadrados = []

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area > 50:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.8 <= aspect_ratio <= 1.2:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    todos_los_cuadrados.append((area, [cX, cY], c, max(w, h)))
    
    todos_los_cuadrados.sort(key=lambda x: x[0], reverse=True)
    anchas_finales = todos_los_cuadrados[:4]

    if len(anchas_finales) != 4: return img_original, None

    pts_list = []
    tamaños_list = []
    for area_sq, centro, contorno, tamaño_lado in anchas_finales:
        pts_list.append(centro)
        tamaños_list.append(tamaño_lado)

    pts = np.array(pts_list, dtype="float32")
    tamaño_promedio = np.mean(tamaños_list)

    img_aplanada = transformar_perspectiva_dinamica(gray, pts, tamaño_promedio)

    # Autocorrección Orientación
    h, w = img_aplanada.shape
    banda = 0.025
    roi_izq = img_aplanada[:, 0:int(w * banda)]
    roi_der = img_aplanada[:, int(w * (1 - banda)):w]
    roi_sup = img_aplanada[0:int(h * banda), :]
    roi_inf = img_aplanada[int(h * (1 - banda)):h, :]

    def densidad_tinta(roi):
        binaria = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return cv2.countNonZero(binaria)

    densidades = {
        "izquierda": densidad_tinta(roi_izq), "derecha": densidad_tinta(roi_der),
        "superior": densidad_tinta(roi_sup), "inferior": densidad_tinta(roi_inf)
    }
    lado_dominante = max(densidades, key=densidades.get)

    if lado_dominante == "superior":
        img_aplanada = cv2.rotate(img_aplanada, cv2.ROTATE_180)
    elif lado_dominante == "izquierda":
        img_aplanada = cv2.rotate(img_aplanada, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif lado_dominante == "derecha":
        img_aplanada = cv2.rotate(img_aplanada, cv2.ROTATE_90_CLOCKWISE)

    return img_original, img_aplanada

def visualizar_y_recortar_zonas(img_aplanada):
    if img_aplanada is None: return None, []
    thresh = cv2.threshold(img_aplanada, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    H, W = img_aplanada.shape
    
    # Coordenadas relativas fijas
    roi_codigo = {'x': int(W * 0.091), 'y': int(H * 0.303), 'w': int(W * (0.338 - 0.091)), 'h': int(H * (0.602 - 0.303))}
    roi_col1 = {'x': int(W * 0.4509), 'y': int(H * 0.065), 'w': int(W * (0.6434 - 0.4509)), 'h': int(H * (0.958 - 0.065))}
    roi_col2 = {'x': int(W * 0.7135), 'y': int(H * 0.065), 'w': int(W * (0.906 - 0.7135)), 'h': int(H * (0.958 - 0.065))}

    return thresh, [roi_codigo, roi_col1, roi_col2]

def evaluar_bloque(imagen_thresh_roi, preguntas_totales=30, opciones_totales=5):
    (H, W) = imagen_thresh_roi.shape
    respuestas_detectadas = []
    step_y = H / preguntas_totales 
    step_x = W / opciones_totales 
    UMBRAL_PRESENCIA = 180 

    for i in range(preguntas_totales):
        y_start = int(i * step_y); y_end = int((i + 1) * step_y)
        fila_contadores = [] 
        for j in range(opciones_totales):
            x_start = int(j * step_x); x_end = int((j + 1) * step_x)
            celda = imagen_thresh_roi[y_start:y_end, x_start:x_end]
            fila_contadores.append(cv2.countNonZero(celda))
        
        burbujas_marcadas = [idx for idx, val in enumerate(fila_contadores) if val > UMBRAL_PRESENCIA]

        if len(burbujas_marcadas) == 0: respuestas_detectadas.append(-1)
        elif len(burbujas_marcadas) == 1: respuestas_detectadas.append(burbujas_marcadas[0])
        else: respuestas_detectadas.append(-2)

    return respuestas_detectadas

def leer_codigo_estudiante(imagen_thresh_roi, columnas_digitos=8, filas_valores=10):
    (H, W) = imagen_thresh_roi.shape
    step_x = W / columnas_digitos
    step_y = H / filas_valores
    codigo_resultante = ""
    UMBRAL_PRESENCIA = 180 

    for j in range(columnas_digitos):
        x_start = int(j * step_x); x_end = int((j + 1) * step_x)
        col_scores = [] 
        for i in range(filas_valores):
            y_start = int(i * step_y); y_end = int((i + 1) * step_y)
            celda = imagen_thresh_roi[y_start:y_end, x_start:x_end]
            col_scores.append(cv2.countNonZero(celda))
        
        indices_marcados = [idx for idx, val in enumerate(col_scores) if val > UMBRAL_PRESENCIA]
        if len(indices_marcados) == 0: codigo_resultante += "?"
        elif len(indices_marcados) == 1: codigo_resultante += str(indices_marcados[0])
        else: return "INVALIDO"

    return codigo_resultante

def procesar_imagen_examen(ruta_imagen):
    _, img_warp = detectar_anchas_y_transformar(ruta_imagen)
    if img_warp is None: return "ERROR_ANCLAS", []

    img_bin, rois = visualizar_y_recortar_zonas(img_warp)
    if not rois: return "ERROR_PROCESAMIENTO", []
    
    roi_c, roi_p1, roi_p2 = rois[0], rois[1], rois[2]

    # Codigo
    crop_c = img_bin[roi_c['y']:roi_c['y']+roi_c['h'], roi_c['x']:roi_c['x']+roi_c['w']]
    codigo_str = leer_codigo_estudiante(crop_c)
    if codigo_str == "INVALIDO": return "INVALIDO", []

    # Preguntas
    crop_p1 = img_bin[roi_p1['y']:roi_p1['y']+roi_p1['h'], roi_p1['x']:roi_p1['x']+roi_p1['w']]
    res_1 = evaluar_bloque(crop_p1, preguntas_totales=30)
    
    crop_p2 = img_bin[roi_p2['y']:roi_p2['y']+roi_p2['h'], roi_p2['x']:roi_p2['x']+roi_p2['w']]
    res_2 = evaluar_bloque(crop_p2, preguntas_totales=30)

    letras = ['A', 'B', 'C', 'D', 'E']
    lista_final = []
    
    for r in res_1 + res_2:
        if r == -1: lista_final.append("")
        elif r == -2: lista_final.append("ANULADA")
        else: lista_final.append(letras[r])

    return codigo_str, lista_final