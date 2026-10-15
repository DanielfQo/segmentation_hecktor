from pathlib import Path
import shutil
import argparse


DEFAULT_TARGET = Path(
    r"C:/Users/danie/OneDrive/Documentos/Universidad/petct/HECKTOR 2025 Training Data"
)


def eliminar_mda(carpeta: Path, rango_inicio: int = 402, rango_fin: int = 447):
    """
    Elimina carpetas cuyos nombres empiezan por MDA-<número>
    dentro del rango indicado.
    """

    carpeta = Path(carpeta)

    if not carpeta.is_dir():
        print(f"Carpeta no encontrada: {carpeta}")
        return

    for n in range(rango_inicio, rango_fin + 1):
        prefijo = f"MDA-{n}"

        for p in carpeta.glob(prefijo + "*"):
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                    print(f"Eliminado: {p}")
            except Exception as e:
                print(f"Error eliminando {p}: {e}")


def limpiar_rtdose(carpeta: Path):
    """
    Entra en cada subcarpeta del training data y elimina:

    1. El archivo exacto:
       <nombre_carpeta>__RTDOSE.nii.gz

    2. Si quedan mas de 3 archivos .nii.gz,
       elimina los extras ordenados por nombre,
       manteniendo los 3 primeros.
    """

    carpeta = Path(carpeta)

    if not carpeta.is_dir():
        print(f"Carpeta no encontrada: {carpeta}")
        return

    for sub in carpeta.iterdir():
        if not sub.is_dir():
            continue

        nombre = sub.name

        rtdose_file = sub / f"{nombre}__RTDOSE.nii.gz"

        if rtdose_file.exists():
            try:
                rtdose_file.unlink()
                print(f"Eliminado RTDOSE: {rtdose_file}")
            except Exception as e:
                print(f"Error eliminando {rtdose_file}: {e}")

        nii_files = sorted(
            [p for p in sub.glob("*.nii.gz") if p.is_file()]
        )

        if len(nii_files) > 3:
            extras = nii_files[3:]

            for extra in extras:
                try:
                    extra.unlink()
                    print(f"Eliminado extra: {extra}")
                except Exception as e:
                    print(f"Error eliminando {extra}: {e}")


def mostrar_resumen(carpeta: Path):
    """
    Muestra cuantos archivos .nii.gz hay en cada subcarpeta.
    No elimina nada.
    """

    carpeta = Path(carpeta)

    if not carpeta.is_dir():
        print(f"Carpeta no encontrada: {carpeta}")
        return

    for sub in carpeta.iterdir():
        if not sub.is_dir():
            continue

        nii_files = list(sub.glob("*.nii.gz"))
        print(f"{sub.name}: {len(nii_files)} archivos .nii.gz")



def main():
    parser = argparse.ArgumentParser(
        description="Herramientas para limpiar carpetas del dataset HECKTOR."
    )

    parser.add_argument(
        "accion",
        choices=["eliminar_mda", "limpiar_rtdose", "resumen", "todo"],
        help="Acción a ejecutar."
    )

    parser.add_argument(
        "--carpeta",
        type=str,
        default=str(DEFAULT_TARGET),
        help="Ruta de la carpeta principal del dataset."
    )

    parser.add_argument(
        "--inicio",
        type=int,
        default=402,
        help="Número inicial para eliminar carpetas MDA."
    )

    parser.add_argument(
        "--fin",
        type=int,
        default=447,
        help="Número final para eliminar carpetas MDA."
    )

    args = parser.parse_args()

    carpeta = Path(args.carpeta)

    if args.accion == "eliminar_mda":
        eliminar_mda(
            carpeta=carpeta,
            rango_inicio=args.inicio,
            rango_fin=args.fin
        )

    elif args.accion == "limpiar_rtdose":
        limpiar_rtdose(carpeta=carpeta)

    elif args.accion == "resumen":
        mostrar_resumen(carpeta=carpeta)

    elif args.accion == "todo":
        eliminar_mda(
            carpeta=carpeta,
            rango_inicio=args.inicio,
            rango_fin=args.fin
        )
        limpiar_rtdose(carpeta=carpeta)
        mostrar_resumen(carpeta=carpeta)


if __name__ == "__main__":
    main()