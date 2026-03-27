import lightning as L
import pandas as pd
from pathlib import Path
import zipfile

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from geosave_engine.tasks.semseg.common.transform import TransformsCompose
from ..common.metadata import MetadataInterpreter
from .dataset import BaseDataset
from ..common.utils import extract_id

IMAGE_EXTENSIONS = ['tif', 'tiff', 'jpg', 'jpeg', 'png', 'bmp', 'jp2']

class TrainDataModule(L.LightningDataModule):
    def __init__(self, 
                 data_dir,
                 metadata_dict,
                 transform_dict,
                 train_kwargs=None,
                 val_kwargs=None,
                 test_kwargs=None,
        ):
        assert sum(metadata_dict['data_split_ratio']) == 1.0, "data_split_ratio must sum to 1.0"

        super().__init__()
        self.save_hyperparameters(ignore=["data_dir", "image_dir", "label_dir"])  
        
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / metadata_dict['split_dir']
        self.metadata_dict = metadata_dict
        self.transform_dict = transform_dict
        self.image_dir = metadata_dict['image_dir']
        self.label_dir = metadata_dict['label_dir']
        self.seed = metadata_dict['split_seed']
        self.overwrite_splits = metadata_dict['overwrite_splits']

        self.train_kwargs = train_kwargs or {}
        self.val_kwargs = val_kwargs or {}
        self.test_kwargs = test_kwargs or {}

        self.data_split_ratio = metadata_dict['data_split_ratio']
        self.train_ratio = self.data_split_ratio[0] 
        self.val_ratio = self.data_split_ratio[1] 
        self.test_ratio = self.data_split_ratio[2]
        
        self.input_size = metadata_dict['input_size']

    def data_split(self):
        images, labels = [], []
        
        img_path_obj = self.data_dir / self.image_dir
        lbl_path_obj = self.data_dir / self.label_dir
        
        for ext in IMAGE_EXTENSIONS:
            # Using rglob allows finding images even if they are in subfolders
            images.extend(list(img_path_obj.rglob(f"*.{ext}")))
            labels.extend(list(lbl_path_obj.rglob(f"*.{ext}")))
        
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
            raise ValueError("No matching image-mask pairs found. Check that filenames match.")
        
        df = pd.DataFrame(labeled_pairs, columns=['image', 'label'])
        unlabeled_df = pd.DataFrame(unlabeled_imgs, columns=['image'])

        train_df, temp_df = train_test_split(
            df, 
            train_size=self.train_ratio, 
            random_state=self.seed, # [NEW] Use instance seed
            shuffle=True
        )
        
        val_df, test_df = train_test_split(
            temp_df, 
            train_size=self.val_ratio / (self.val_ratio + self.test_ratio),
            random_state=self.seed, # [NEW] Use instance seed
            shuffle=True
        )
        
        return train_df, val_df, test_df, unlabeled_df
    
    def prepare_data(self): 
        zip_path = Path(self.metadata_dict['data_zip'])
        extract_dir = Path(self.metadata_dict['data_dir'])
        
        if not extract_dir.exists():
            print(f"Extracting {zip_path.name} to {extract_dir}...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

        self.split_dir.mkdir(exist_ok=True) 
        is_split_exists = all((self.split_dir / f"{split}_split.csv").exists() for split in ["train", "val", "test", "unlabeled"])
        
        if not is_split_exists or self.overwrite_splits:
            print("Creating new data splits...")
            train_df, val_df, test_df, unlabeled_df = self.data_split()
            train_df.to_csv(self.split_dir / "train_split.csv", index=False)
            val_df.to_csv(self.split_dir / "val_split.csv", index=False)
            test_df.to_csv(self.split_dir / "test_split.csv", index=False)
            unlabeled_df.to_csv(self.split_dir / "unlabeled_split.csv", index=False)

    def setup(self, stage=None):
        transform_trn = TransformsCompose(self.transform_dict['train'], input_size=self.input_size)
        transform_infer = TransformsCompose(self.transform_dict['infer'], input_size=self.input_size)
        metadata = MetadataInterpreter(self.metadata_dict)
        
        if stage == "fit" or stage is None:
            train_df = pd.read_csv(self.split_dir / "train_split.csv")
            val_df = pd.read_csv(self.split_dir / "val_split.csv")
            self.train_ds = BaseDataset(self.data_dir, train_df, metadata, transform_trn, self.image_dir, self.label_dir)
            self.val_ds = BaseDataset(self.data_dir, val_df, metadata, transform_infer, self.image_dir, self.label_dir)
        
        elif stage == "validate":
            val_df = pd.read_csv(self.split_dir / "val_split.csv")
            self.val_ds = BaseDataset(self.data_dir, val_df, metadata, transform_infer, self.image_dir, self.label_dir)

        elif stage == "test":
            test_df = pd.read_csv(self.split_dir / "test_split.csv")
            self.test_ds = BaseDataset(self.data_dir, test_df, metadata, transform_infer, self.image_dir, self.label_dir)
        
        else:
            raise ValueError(f"Unknown stage: {stage}. Expected 'fit', 'validate', or 'test'.")

    def train_dataloader(self):
        default_kwargs = {
            'batch_size': 16,
            'shuffle': True,
            'drop_last': True,
            'num_workers': 4,      # [NEW] Default workers for speed
            'pin_memory': True     # [NEW] Speeds up CPU -> GPU memory transfer
        }
        default_kwargs.update(self.train_kwargs)
        return DataLoader(self.train_ds, **default_kwargs)

    def val_dataloader(self):
        default_kwargs = {
            'batch_size': 16,
            'shuffle': False,
            'drop_last': False,
            'num_workers': 4,
            'pin_memory': True
        }
        default_kwargs.update(self.val_kwargs)
        return DataLoader(self.val_ds, **default_kwargs)

    def test_dataloader(self):
        default_kwargs = {
            'batch_size': 16,
            'shuffle': False,
            'drop_last': False,
            'num_workers': 4,
            'pin_memory': True
        }
        default_kwargs.update(self.test_kwargs)
        return DataLoader(self.test_ds, **default_kwargs)
    

class PredictDataModule(L.LightningDataModule):
    def __init__(self, 
                 predict_dir: str, 
                 metadata_dict: dict,  # Extracted directly from the loaded model!
                 transforms_dict: dict, # Extracted directly from the loaded model!
                 predict_kwargs=None
        ):
        super().__init__()
        self.predict_dir = Path(predict_dir)
        self.metadata_dict = metadata_dict
        self.transforms_dict = transforms_dict
        self.predict_kwargs = predict_kwargs or {}

    def setup(self, stage=None):
        if stage != "predict" and stage is not None:
            raise ValueError(f"PredictDataModule only supports 'predict' stage, got '{stage}'")
        
        images = [img.name for img in self.predict_dir.rglob("*.*") if img.suffix in IMAGE_EXTENSIONS]
        
        if not images:
            raise FileNotFoundError(f"No images found in {self.predict_dir}")
            
        predict_df = pd.DataFrame({'image': images})
        
        self.predict_ds = BaseDataset(
            data_dir=self.predict_dir, 
            data_df=predict_df, 
            metadata_cls=self.metadata_dict, 
            transform_fn=self.transforms_dict['infer'], 
            image_dir="", # Files are directly in predict_dir
            label_dir=None, # CRITICAL: Tell the dataset not to look for masks
            predict_mode=True
        )

    def predict_dataloader(self):
        default_kwargs = {
            'batch_size': 16,
            'shuffle': False,
            'drop_last': False,
            'num_workers': 4,
            'pin_memory': True
        }
        default_kwargs.update(self.predict_kwargs)
        return DataLoader(
            self.predict_ds, 
            **default_kwargs
        )