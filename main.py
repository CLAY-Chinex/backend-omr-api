from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import backend_omr # Importamos tu cerebro profesional

app = FastAPI()

@app.get("/")
def home():
    return {"mensaje": "Servidor OMR Profesional Activo"}

@app.post("/procesar_examen")
async def procesar_examen(file: UploadFile = File(...)):
    # 1. Guardar temporalmente la imagen que llega del celular
    temp_filename = f"temp_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Usar el Cerebro (backend_omr)
        # Paso A: Procesar imagen y obtener binaria corregida
        img_bin, mensaje = backend_omr.processor.procesar_imagen(temp_filename)
        
        if img_bin is None:
            # Si falló la detección de la hoja
            raise HTTPException(status_code=400, detail=f"Error visión: {mensaje}")
            
        # Paso B: Extraer respuestas (1-60)
        codigo, respuestas = backend_omr.processor.analizar_examen_completo(img_bin)
        
        # 3. Responder al celular (JSON)
        return {
            "status": "success",
            "codigo_estudiante": codigo,
            "respuestas_detectadas": respuestas # Lista de 60 items
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 4. Limpieza siempre (borrar archivo temporal)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)