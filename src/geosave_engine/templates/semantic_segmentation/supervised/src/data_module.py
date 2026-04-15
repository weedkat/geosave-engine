from lightning import LightningDataModule

class GeosaveDataModule(LightningDataModule):
    """
    Base class for GeoSave data modules. 
    Data modules are responsible for loading and preprocessing data for training and inference.
    They should implement the `prepare_data` and `setup` methods to handle data loading and splitting.
    """
    pass