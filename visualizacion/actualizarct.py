from pathlib import Path
import shutil

origen = Path(r"C:\Users\danie\OneDrive\Documentos\Universidad\petct\HECKTOR 2025 Training Data Defaced New CT")
destino = Path(r"C:\Users\danie\OneDrive\Documentos\Universidad\petct\HECKTOR 2025 Training Data")

for carpeta_paciente in origen.iterdir():
    if not carpeta_paciente.is_dir():
        continue

    paciente = carpeta_paciente.name
    carpeta_destino = destino / paciente

    if not carpeta_destino.exists():
        print(f"No existe: {paciente}")
        continue

    for archivo in carpeta_paciente.glob("*.nii.gz"):
        destino_archivo = carpeta_destino / archivo.name

        shutil.copy2(archivo, destino_archivo)

        print(f"Actualizado: {paciente}/{archivo.name}")

print("Proceso terminado")