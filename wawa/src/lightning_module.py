from lightning import LightningModule
from src.factory import build_model  # ty:ignore[unresolved-import]


class GeosaveLightningModule(LightningModule):
    """
    Base class for GeoSave lightning modules. 
    Lightning modules are responsible for defining the neural network architecture and the training logic.
    They should implement the `training_step`, `validation_step`, and `test_step` methods to handle the training, validation, and testing loops.
    """
    def __init__(self, model_config, optimzer_config, loss_config):
        super().__init__()
        self.save_hyperparameters()
        model_name = model_config.get("name") if isinstance(model_config, dict) else None
        model_kwargs = model_config.get("model_config", {}) if isinstance(model_config, dict) else {}
        if model_name is None:
            raise ValueError("model.name is required in config to build a model instance.")
        if not isinstance(model_kwargs, dict):
            raise ValueError("model.model_config must be a dictionary.")
        self.model = build_model(model_name, **model_kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        raise NotImplementedError("The configure_optimizers method is not used in the data module. This class is responsible for data loading and preprocessing, not model training.")

    def training_step(self, *args, **kwargs):
        raise NotImplementedError("The training_step method is not used in the data module. This class is responsible for data loading and preprocessing, not model training.")
    
    def on_train_epoch_end(self, *args, **kwargs):
        pass  # No-op, data module does not handle epoch end logic for training

    def validation_step(self, *args, **kwargs):
        raise NotImplementedError("The validation_step method is not used in the data module. This class is responsible for data loading and preprocessing, not model training.")
    
    def on_validation_epoch_end(self, *args, **kwargs):
        pass  # No-op, data module does not handle epoch end logic for validation

    def test_step(self, *args, **kwargs):
        raise NotImplementedError("The test_step method is not used in the data module. This class is responsible for data loading and preprocessing, not model training.")
    
    def on_test_epoch_end(self, *args, **kwargs):
        pass  # No-op, data module does not handle epoch end logic for testing

    def predict_step(self, *args, **kwargs):
        raise NotImplementedError("The predict_step method is not used in the data module. This class is responsible for data loading and preprocessing, not model inference.")
    
    def on_predict_epoch_end(self, *args, **kwargs):
        pass  # No-op, data module does not handle epoch end logic for prediction



