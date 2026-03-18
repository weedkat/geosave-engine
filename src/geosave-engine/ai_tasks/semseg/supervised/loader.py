import lightning as L
import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .dataset import SemSegDataset

from ..core.utils import extract_id

SEED = 10
IMAGE_EXTENSIONS = ['tif', 'tiff', 'jpg', 'jpeg', 'png', 'bmp', 'jp2']

class DataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, data_csv=None, image_dir='images', label_dir='labels'):
        super().__init__()
        self.save_hyperparameters()  # Saves the arguments to self.hparams
        self.data_dir = Path(data_dir)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.data_csv = data_csv

    def data_split(self):
        if self.data_csv is None:
            images, labels = [], []
            for ext in IMAGE_EXTENSIONS:
                image_files = list(self.data_dir.glob(f"{self.image_dir}/*.{ext}"))
                label_files = list(self.data_dir.glob(f"{self.label_dir}/*.{ext}"))
                images.extend(image_files)
                labels.extend(label_files)
            
            images = sorted(images)
            labels = sorted(labels)
    
            label_dict = {extract_id(label.name): label.name for label in labels}

              # Match images to masks by ID
            labeled_pairs = []
            unlabeled_imgs = []
            
            for img in images:
                img_name = img.name
                img_id = extract_id(img_name)
                if img_id in label_dict:
                    label_name = label_dict[img_id]
                    labeled_pairs.append((img_name, label_name))
                else:
                    unlabeled_imgs.append(img_name)
            
            if not labeled_pairs:
                raise ValueError(f"No matching image-mask pairs found. Check that filenames match (ignoring extensions)")
            
            df = pd.DataFrame(labeled_pairs, columns=['image', 'label'])
            unlabeled_df = pd.DataFrame(unlabeled_imgs, columns=['image'])

        else:
            df = pd.read_csv(self.data_csv)
            unlabeled_df = None  # No unlabeled data when using a CSV
        
        train_df, temp_df = train_test_split(
            df, 
            train_size=0.7, 
            random_state=42, 
            shuffle=True
        )
        
        # 2. Split (Val + Test) exactly in half
        val_df, test_df = train_test_split(
            temp_df, 
            train_size=0.5, 
            random_state=42, 
            shuffle=True
        )
        
        # Unlabeled data remains separate
        return train_df, val_df, test_df, unlabeled_df
    

    def prepare_data(self): 
        # Only called on 1 GPU/TPU in distributed mode
        is_split_exists = all((self.data_dir / f"{split}_split.csv").exists() for split in ["train", "val", "test"])
        
        if not is_split_exists:
            # Create new splits and save them
            train_df, val_df, test_df, unlabeled_df = self.data_split()
            train_df.to_csv(self.data_dir / "train_split.csv", index=False)
            val_df.to_csv(self.data_dir / "val_split.csv", index=False)
            test_df.to_csv(self.data_dir / "test_split.csv", index=False)
            if unlabeled_df is not None:
                unlabeled_df.to_csv(self.data_dir / "unlabeled_data.csv", index=False)
        

    def setup(self, stage=None):
        # This runs on EVERY GPU. 
        if stage == "fit":
            train_df = pd.read_csv("train_split.csv")
            val_df = pd.read_csv("val_split.csv")
            
            self.train_ds = SemSegDataset(train_df)
            self.val_ds = SemSegDataset(val_df)
        
        if stage == "validate":
            val_df = pd.read_csv("val_split.csv")
            self.val_ds = SemSegDataset(val_df)

        elif stage == "test":
            test_df = pd.read_csv("test_split.csv")
            self.test_ds = SemSegDataset(test_df)
        
        elif stage == "predict":
            predict_df = pd.read_csv("predict_split.csv")
            self.predict_ds = SemSegDataset(predict_df)
        
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return super().test_dataloader()
    
    def predict_dataloader(self):
        return super().predict_dataloader()