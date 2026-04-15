from lightning import LightningDataModule


class GeosaveDataModule(LightningDataModule):
    """
    Base class for GeoSave data modules. 
    Data modules are responsible for loading and preprocessing data for training and inference.
    They should implement the `prepare_data` and `setup` methods to handle data loading and splitting.
    """
    def __init__(self, data_dir, train_kwargs, val_kwargs):
        super().__init__()
        self.save_hyperparameters()
    
    def data_split(self):
        pass

    def prepare_data(self):
        # This method is called only on 1 GPU - good for downloading data, tokenization, etc.
        pass  # No-op, data module does not handle data preparation logic in this template

    def setup(self, stage=None):
        # This method is called on every GPU separately - stage defines if we are at fit or test step
        if stage == 'fit' or stage is None:
            pass

        elif stage == 'validate':
            pass
        
        elif stage == 'test':
            pass
        
        elif stage == 'predict':
            pass

        else:
            raise ValueError(f"Unknown stage: {stage}. Expected one of 'fit', 'validate', 'test', or 'predict'.")
    
    def train_dataloader(self):
        raise NotImplementedError("The train_dataloader method is not implemented. This class is responsible for data loading and preprocessing, but the specific dataloader implementation is left to the user.")
    
    def val_dataloader(self):
        raise NotImplementedError("The val_dataloader method is not implemented. This class is responsible for data loading and preprocessing, but the specific dataloader implementation is left to the user.")
    
    def test_dataloader(self):
        raise NotImplementedError("The test_dataloader method is not implemented. This class is responsible for data loading and preprocessing, but the specific dataloader implementation is left to the user.")
    
    def predict_dataloader(self):
        raise NotImplementedError("The predict_dataloader method is not implemented. This class is responsible for data loading and preprocessing, but the specific dataloader implementation is left to the user.")
    
    def _predict_setup(self):
        # Optional method to set up a separate predict dataset if needed
        pass  # No-op, data module does not handle predict dataset setup in this template



