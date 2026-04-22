from torchgeo.datasets import RasterDataset

class GeosaveDataset(RasterDataset):
    """Dataset base for semantic segmentation tasks.

    Implement the TorchGeo indexing and sample pairing logic in a project-specific
    subclass once the imagery/label contract is known.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the dataset with the given root directory and split.
        
        Args:
            root (str): Root directory where the dataset is stored.
            split (str): Dataset split to use ("train", "val", "test"). Default is "train".
            **kwargs: Additional keyword arguments for dataset initialization.
        """
    def __getitem__(self, index):
        """Retrieve a single geospatial sample.

        The exact return structure is intentionally undefined in this template.
        """
        raise NotImplementedError(
            "GeosaveDataset.__getitem__ must be implemented once the batch contract is defined."
        )
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        raise NotImplementedError("GeosaveDataset.__len__ must be implemented once the dataset structure is defined.")
