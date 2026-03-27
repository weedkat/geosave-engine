import yaml
from .module import SemSegModel
from .loader import TrainDataModule
import lightning as L

def train(metadata_yaml, transform_yaml, train_yaml):
    with open(metadata_yaml) as f:
        metadata_dict = yaml.safe_load(f)
    with open(transform_yaml) as f:
        transform_dict = yaml.safe_load(f)
    with open(train_yaml) as f:
        train_dict = yaml.safe_load(f)

    dm_train = TrainDataModule(
        data_dir=metadata_dict['data_dir'],
        metadata_dict=metadata_dict,
        transform_dict=transform_dict,
        train_kwargs=train_dict['train_loader_cfg'],
        val_kwargs=train_dict['eval_loader_cfg'],
        test_kwargs=train_dict['eval_loader_cfg'],
    )

    model = SemSegModel(
        arch_cfg=train_dict['arch_cfg'],
        optim_cfg=train_dict['optim_cfg'],
        loss_cfg=train_dict['loss_cfg'],
        metadata_dict=metadata_dict,
        transform_dict=transform_dict,
    )

    train_config = train_dict['train_cfg']

    trainer = L.Trainer(
        **train_config,
        logger=L.loggers.TensorBoardLogger("tb_logs", name="semseg"),
        callbacks=[
            L.callbacks.ModelCheckpoint(monitor="val/mIoU", mode="max", save_top_k=3),
            L.callbacks.EarlyStopping(monitor="val/mIoU", mode="max", patience=10, verbose=True)
        ]
    )

    trainer.fit(model, dm_train)

    

    