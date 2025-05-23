import os
import pandas as pd

# Ruta del directorio con los CSV
input_dir = r'C:\Users\marcos\Downloads\LoRA2'

output_dir =r'C:\Users\marcos\Downloads\LoRA2'

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Procesar todos los archivos .csv en el directorio
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        csv_path = os.path.join(input_dir, filename)
        json_path = os.path.join(output_dir, filename.replace('.csv', '.json'))

        # Leer el CSV
        df = pd.read_csv(csv_path)

        # Convertir y guardar como JSON
        df.to_json(json_path, orient='records', lines=False, force_ascii=False, indent=4)

        print(f'Convertido: {filename} → {os.path.basename(json_path)}')

print("Conversión completa.")
