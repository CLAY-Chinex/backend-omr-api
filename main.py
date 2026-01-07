from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import uuid
import logging
import backend_omr  # Tu módulo con la nueva clase OMRProcessor

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
# Configuración de logs para ver peticiones en el dashboard de Render
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("API_OMR")

app = FastAPI(title="API OMR Grading", version="2.0")

@app.get("/")
def home():
    return {"mensaje": "API OMR v2.0 - Sistema Operativo y Listo"}

@app.post("/procesar_examen")
async def procesar_examen(file: UploadFile = File(...)):
    """
    Endpoint principal: Recibe una imagen, la procesa y devuelve las notas.
    """
    # 1. Generar un nombre de archivo único para evitar colisiones entre usuarios
    # uuid4 genera una cadena aleatoria única (ej: 550e8400-e29b...)
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"temp_{uuid.uuid4()}{file_extension}"
    
    logger.info(f"Recibiendo archivo: {file.filename} -> Guardado como: {unique_filename}")

    try:
        # 2. Guardar la imagen en disco
        with open(unique_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Procesar con la lógica del backend actualizado
        # La función procesar_imagen_examen actúa como puente con tu clase OMRProcessor
        codigo_resultado, respuestas = backend_omr.procesar_imagen_examen(unique_filename)
        
        logger.info(f"Resultado procesamiento: {codigo_resultado}")

        # 4. Manejo de Errores Específicos (Mapeo Backend -> HTTP)
        if codigo_resultado == "ERROR_LECTURA_ARCHIVO":
            raise HTTPException(status_code=400, detail="El archivo subido no es una imagen válida o está corrupto.")
            
        if codigo_resultado == "ERROR_ANCLAS":
            raise HTTPException(
                status_code=422, 
                detail="No se detectaron los 4 cuadrados guía (anclas). Asegúrate de que la hoja esté plana y con buena luz."
            )
            
        if codigo_resultado == "ERROR_PROCESAMIENTO":
            raise HTTPException(status_code=500, detail="Error interno al procesar la imagen.")

        if codigo_resultado == "INVALIDO":
            raise HTTPException(
                status_code=422, 
                detail="El código del estudiante no se pudo leer correctamente (marca doble o vacía en la columna)."
            )

        # 5. Respuesta Exitosa
        return {
            "status": "success",
            "archivo_procesado": file.filename,
            "data": {
                "codigo_estudiante": codigo_resultado,
                "respuestas_detectadas": respuestas
            }
        }
        
    except HTTPException as he:
        # Re-lanzar las excepciones HTTP controladas
        raise he
        
    except Exception as e:
        # Capturar cualquier error no previsto (bugs)
        logger.error(f"Error crítico no controlado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
        
    finally:
        # 6. LIMPIEZA OBLIGATORIA
        # Este bloque se ejecuta SIEMPRE, haya error o no.
        if os.path.exists(unique_filename):
            os.remove(unique_filename)
            logger.info(f"Archivo temporal eliminado: {unique_filename}")

if __name__ == "__main__":
    import uvicorn
    # En Render, el puerto lo asigna la variable de entorno PORT, pero uvicorn lo maneja
    uvicorn.run(app, host="0.0.0.0", port=8000)