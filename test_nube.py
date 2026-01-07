import requests

# 1. PEGA AQUÍ TU URL DE RENDER (¡Sin el /docs al final!)
url_api = "https://api-omr-movil.onrender.com/procesar_examen"  
# (Asegúrate de cambiar lo de arriba por TU url real)

# 2. Ruta de una imagen de prueba en tu PC
ruta_imagen = "optica.png"  # <--- CAMBIA ESTO por el nombre de tu foto real

print("⏳ Enviando imagen a la nube... (Esto puede tardar unos segundos)")

try:
    # Preparamos el archivo
    with open(ruta_imagen, 'rb') as f:
        archivos = {'file': f}
        
        # Hacemos la petición POST (igual que hará el celular)
        respuesta = requests.post(url_api, files=archivos)
    
    # Verificamos si salió bien
    if respuesta.status_code == 200:
        datos = respuesta.json()
        print("\n✅ ¡ÉXITO! El servidor respondió:")
        print(f"Código Estudiante: {datos['codigo_estudiante']}")
        print(f"Respuestas: {datos['respuestas_detectadas']}")
    else:
        print("\n❌ Error en el servidor:")
        print(respuesta.text)

except FileNotFoundError:
    print(f"❌ Error: No encuentro la imagen '{ruta_imagen}' en esta carpeta.")
except Exception as e:
    print(f"❌ Error de conexión: {e}")