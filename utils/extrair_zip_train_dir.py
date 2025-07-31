import zipfile
import os
from pathlib import Path
from tqdm import tqdm

def descompactar_zip_com_progresso(zip_path, destino="./train"):
    """
    Extrai um arquivo .zip contendo arquivos .pt para a pasta ./train com barra de progresso.
    Aborta se a pasta de destino já contiver arquivos.

    Parâmetros:
        zip_path (str ou Path): Caminho para o arquivo .zip
        destino (str ou Path): Pasta de destino da extração (default: ./train)
    """
    zip_path = Path(zip_path)
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    # Verificação: pasta não deve conter arquivos
    if any(destino.iterdir()):
        print(f"⚠️ A pasta '{destino}' já contém arquivos. A extração foi abortada para evitar sobrescrita.")
        return

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        arquivos = zip_ref.namelist()
        arquivos_pt = [f for f in arquivos if f.endswith(".pt")]

        if not arquivos_pt:
            print("❌ Nenhum arquivo .pt encontrado no .zip.")
            return

        print(f"Extraindo {len(arquivos_pt)} arquivos .pt para '{destino}'...")

        for file in tqdm(arquivos_pt, desc="Extraindo", unit="arquivo"):
            zip_ref.extract(file, destino)

    print("✅ Extração concluída com sucesso.")
