import lightning as L
import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from geosave_engine.ai_tasks.semseg.supervised.metadata import MetadataInterpreter
from .dataset import SemSegDataset

from ..core.utils import extract_id

SEED = 10
IMAGE_EXTENSIONS = ['tif', 'tiff', 'jpg', 'jpeg', 'png', 'bmp', 'jp2']

class DataModule(L.LightningDataModule):
    def __init__(self, 
                 data_dir, 
                 data_csv=None,
                 metadata=None, 
                 image_dir='images', 
                 label_dir='labels', 
                 data_split_ratio=(0.7, 0.15, 0.15), 
                 **loader_kwargs
        ):
        assert sum(data_split_ratio) == 1.0, "data_split_ratio must sum to 1.0"

        super().__init__()
        self.save_hyperparameters()  # Saves the arguments to self.hparams
        
        self.data_dir = Path(data_dir)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.data_csv = data_csv
        self.loader_kwargs = loader_kwargs
        
        if metadata is not None:
            self.metadata = MetadataInterpreter(metadata).metadata

        self.train_ratio, self.val_ratio, self.test_ratio = data_split_ratio


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
            is_columns_valid = all(col in pd.read_csv(self.data_csv, nrows=0).columns for col in ['image', 'label'])
            if not is_columns_valid:
                raise ValueError(f"Invalid column names in {self.data_csv}. Expected columns: ['image', 'label']")
            
            df = pd.read_csv(self.data_csv)

            df['image'] = df['image'].apply(lambda x: Path(x).name) # Ensure only filename is used, not full path.
            df['label'] = df['label'].apply(lambda x: Path(x).name if pd.notna(x) else x)
            
            unlabeled_df = df[df['label'].isna()]
        

        train_df, temp_df = train_test_split(
            df, 
            train_size=self.train_ratio, 
            random_state=SEED, 
            shuffle=True
        )
        
        val_df, test_df = train_test_split(
            temp_df, 
            train_size=self.val_ratio / (self.val_ratio + self.test_ratio),
            random_state=SEED, 
            shuffle=True
        )
        
        return train_df, val_df, test_df, unlabeled_df
    

    def prepare_data(self): 
        # Only called on 1 GPU/TPU in distributed mode
        is_split_exists = all((self.data_dir / f"{split}_split.csv").exists() for split in ["train", "val", "test"])
        
        if not is_split_exists:
            print("Split csv files not found. Creating new splits...")
            train_df, val_df, test_df, unlabeled_df = self.data_split()
            train_df.to_csv(self.data_dir / "train_split.csv", index=False)
            val_df.to_csv(self.data_dir / "val_split.csv", index=False)
            test_df.to_csv(self.data_dir / "test_split.csv", index=False)
            unlabeled_df.to_csv(self.data_dir / "unlabeled_data.csv", index=False)
        

    def setup(self, stage=None):
        # Reach into the model to see if it already has the 'truth'
        if self.trainer and self.trainer.model and hasattr(self.trainer.model, "metadata"):
            self.metadata = self.trainer.model.metadata

        # This runs on EVERY GPU. 
        if stage == "fit":
            train_df = pd.read_csv(self.data_dir / "train_split.csv")
            val_df = pd.read_csv(self.data_dir / "val_split.csv")
            
            self.train_ds = SemSegDataset(self.data_dir, train_df, self.metadata, self.image_dir, self.label_dir)
            self.val_ds = SemSegDataset(self.data_dir, val_df, self.metadata, self.image_dir, self.label_dir)
        
        elif stage == "validate":
            val_df = pd.read_csv(self.data_dir / "val_split.csv")
            self.val_ds = SemSegDataset(self.data_dir, val_df, self.metadata, self.image_dir, self.label_dir)

        elif stage == "test":
            test_df = pd.read_csv(self.data_dir / "test_split.csv")
            self.test_ds = SemSegDataset(self.data_dir, test_df, self.metadata, self.image_dir, self.label_dir)
        
        elif stage == "predict":
            images = []
            for ext in IMAGE_EXTENSIONS:
                image_files = list(self.data_dir.glob(f"{self.image_dir}/*.{ext}"))
                images.extend(image_files)
            predict_df = pd.DataFrame({'image': images})
            self.predict_ds = SemSegDataset(self.data_dir, predict_df, self.metadata_yaml, self.image_dir, self.label_dir)
        

    def train_dataloader(self):
        default_kwargs = {
            'batch_size': 32,
            'num_workers': 4,
            'shuffle': True,
            'drop_last': True,
        }
        default_kwargs.update(self.loader_kwargs)
        return DataLoader(self.train_ds, **default_kwargs)

    def val_dataloader(self):
        default_kwargs = {
            'batch_size': 32,
            'num_workers': 2,
            'shuffle': False,
        }
        default_kwargs.update(self.loader_kwargs)
        return DataLoader(self.val_ds, **default_kwargs)

    def test_dataloader(self):
        default_kwargs = {
            'batch_size': 32,
            'num_workers': 2,
            'shuffle': False,
        }
        default_kwargs.update(self.loader_kwargs)
        return DataLoader(self.test_ds, **default_kwargs)
    
    def predict_dataloader(self):
        default_kwargs = {
            'batch_size': 32,
            'num_workers': 2,
            'shuffle': False,
        }
        default_kwargs.update(self.loader_kwargs)
        return DataLoader(self.predict_ds, **default_kwargs)