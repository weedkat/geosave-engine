from lightning import LightningModule

class GeosaveLightningModule(LightningModule):
    """
    Base class for GeoSave lightning modules. 
    Lightning modules are responsible for defining the neural network architecture and the training logic.
    They should implement the `training_step`, `validation_step`, and `test_step` methods to handle the training, validation, and testing loops.
    """
    pass