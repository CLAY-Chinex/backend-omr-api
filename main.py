from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import backend_omr # Importamos tu logica

app = FastAPI()

@app.get("/")
def home():
    return {"mensaje": "API OMR Funcionando Correctamente"}

@app.post("/procesar_examen")
async def procesar_examen(file: UploadFile = File(...)):
    # 1. Guardar la imagen recibida temporalmente
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Procesar con tu lógica OMR
        codigo, respuestas = backend_omr.procesar_imagen_examen(temp_filename)
        
        # 3. Limpieza (Borrar archivo temporal)
        os.remove(temp_filename)
        
        # 4. Responder al celular
        if codigo == "ERROR_ANCLAS":
            raise HTTPException(status_code=400, detail="No se detectaron las anclas en la imagen")
        if codigo == "INVALIDO":
            raise HTTPException(status_code=400, detail="Código de estudiante inválido")
            
        return {
            "status": "success",
            "codigo_estudiante": codigo,
            "respuestas_detectadas": respuestas
        }
        
    except Exception as e:
        # En caso de error, intentar borrar el temporal si existe
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)