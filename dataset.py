import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import utils_gan.extrair_zip_train_dir as zipService


class ImageStitchingDatasetFiles(Dataset):
    def __init__(self, folder_path, use_gradiente=False):
        self.folder = Path(folder_path)
        self.use_gradiente = use_gradiente
        # Lista todos arquivos .pt ordenados
        self.files = sorted(self.folder.glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])

        def to_float_tensor(t):
            # uint8 [0..255] -> float32 [0..1]
            return t.float() / 255.0

        parte1 = to_float_tensor(sample["parte1"])
        parte2 = to_float_tensor(sample["parte2"])
        groundtruth = to_float_tensor(sample["groundtruth"])

        if self.use_gradiente:
            gradiente = to_float_tensor(sample["gradiente"])
            return (parte1, parte2), groundtruth, gradiente
        else:
            return (parte1, parte2), groundtruth
