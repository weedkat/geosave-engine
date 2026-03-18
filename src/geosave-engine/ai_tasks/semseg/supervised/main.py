import lightning as L

class SemSegModel(L.LightningModule):
    def __init__(self, arch):
        super().__init__()
        self.model = arch
        self.arch = arch

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)
    
    def predict_step(self, *args, **kwargs):
        return super().predict_step(*args, **kwargs)
    
