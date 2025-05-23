import os
import zipfile

# Ruta del directorio a comprimir
carpeta = r"C:\Users\marcos\Downloads\summit"

# Ruta del archivo ZIP de salida
zip_salida = os.path.join(carpeta, "archivos_json_corregidos.zip")

# Crear ZIP
with zipfile.ZipFile(zip_salida, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for archivo in os.listdir(carpeta):
        ruta_archivo = os.path.join(carpeta, archivo)
        if archivo.endswith('.json') and os.path.isfile(ruta_archivo):
            zipf.write(ruta_archivo, arcname=archivo)  # arcname evita rutas absolutas
            print(f"Añadido al ZIP: {archivo}")

print(f"\n✅ ZIP creado en: {zip_salida}")
