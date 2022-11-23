import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.data.preprocessing import Preprocessor

class BirdCLEFDataset(Dataset):
    def __init__(self, df, node_cfg_dataset, mode="train"):
        self.df = df
        self.mode = mode
        self.preprocessor = Preprocessor(node_cfg_dataset)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_path = sample["file_path"]
        labels = sample["new_target"]
        
        image = self.preprocessor(wav_path, self.mode)
        
        targets = np.zeros(len(CFG.target_columns), dtype=float)
        for ebird_code in labels.split():
            targets[CFG.target_columns.index(ebird_code)] = 1.0

        return image, targets
    
def build_data_loader(df, data_cfg, mode="train") -> DataLoader:
    """
    generate data loader
    :param data_list: list of (img, labels)
    :param data_cfg: data config node/
    :param is_training: whether training
    :return: data loader
    """
    dataset = BirdCLEFDataset(df, data_cfg, mode)
    collator = BirdCLEFDataset(mode, data_cfg.DO_AUGMIX)
    batch_size = data_cfg.BATCH_SIZE

    # limit the number of works based on CPU number.
    num_workers = min(batch_size, data_cfg.CPU_NUM)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training, collate_fn=collator,
                             num_workers=num_workers)
    return 