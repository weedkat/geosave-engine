import argparse

from geomoka.train.trainer import SegmentationTrainer


def main():
    parser = argparse.ArgumentParser(description='GeoMoka training CLI')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--save_dir', type=str, default='output', help='path to save checkpoints and logs')
    args = parser.parse_args()

    trainer = SegmentationTrainer.from_config(args.config, save_dir=args.save_dir)
    trainer.run()

    @classmethod
    def from_config(cls, config_path: Union[str, Path], save_dir: str = 'outputs'):
        """
        Load trainer from config YAML file.
        
        Args:
            config_path: Path to config YAML
            save_dir: Base directory for saving outputs
            
        Returns:
            SegmentationTrainer instance
        """
        # Load configuration from YAML file
        cfg = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        metadata = yaml.load(open(cfg['metadata'], 'r'), Loader=yaml.Loader)
        mc = MetadataInterpreter(metadata)

        train_cfg = cfg['train_cfg']
        model_cfg = cfg['model_cfg']
        transform_cfg = cfg['transform_cfg']
        
        # Generate unique model name and save path
        model_name = generate_model_name(cfg)
        save_path = Path(save_dir) / model_name
        save_path.mkdir(parents=True, exist_ok=True)

        # Save full config
        with open(save_path / 'config.yaml', 'w') as f:
            yaml.safe_dump(cfg, f)

        # ====================== Logger & Tensorboard =========================
        logger = init_log(
            name='global',
            level=logging.INFO,
            log_file=str(save_path / 'training.log'),
            add_console=False,
            rank_filter=True,
        )
        
        print('\n' + '='*80)
        print('Training Configuration:')
        print('='*80)
        print(pprint.pformat(cfg))
        print('='*80 + '\n')
        
        logger.info('{}\n'.format(pprint.pformat(cfg)))
        writer = SummaryWriter(str(save_path))

        # ==================== Model ====================
        print('Building model...')

        model_cfg = {**model_cfg,
            'bands': cfg['bands'],
            'nclass': mc.nclass,
            'in_channels': len(cfg['bands'])
        }
        
        model = SegmentationModel(model_cfg, transform_cfg, metadata)
        print('Model built successfully.')

        # ==================== Optimizer ====================
        encoder_params, decoder_params = model.get_encoder_decoder_params()
        
        optimizer = AdamW(
            [
                {'params': [p for p in encoder_params if p.requires_grad], 'lr': train_cfg['lr']},
                {'params': [p for p in decoder_params if p.requires_grad], 'lr': train_cfg['lr'] * train_cfg['lr_multi']}
            ], 
            lr=train_cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
        )

        # ==================== Criterion ====================
        ignore_index = metadata['ignore_index']
        train_kwargs = train_cfg['criterion'].get('kwargs') or {}

        if train_cfg['criterion']['name'] == 'CELoss':
            criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, **train_kwargs).to(model.device)
        elif train_cfg['criterion']['name'] == 'OHEM':
            criterion = ProbOhemCrossEntropy2d(ignore_index=ignore_index,**train_kwargs).to(model.device)
        else:
            raise NotImplementedError(f"{train_cfg['criterion']['name']} criterion is not implemented")
        
        print(f'Total params: {count_params(model.model):.1f}M\n')
        logger.info(f'Total params: {count_params(model.model):.1f}M\n')

        # ==================== DataLoader ========================
        print('Building dataloaders...')
        trainloader, valloader = None, None
        if train_cfg['method'] == 'supervised':
            trainloader, valloader = cls.build_supervised_dataloader(cfg)
        else:
            raise NotImplementedError(f"{train_cfg['method']} training method not yet implemented")
        print('Dataloaders built successfully.\n')

        return cls(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            trainloader=trainloader,
            valloader=valloader,
            train_cfg=train_cfg,
            save_path=save_path,
            writer=writer,
            logger=logger
        )

    @staticmethod
    def build_supervised_dataloader(cfg):
        """
        Build supervised training and validation dataloaders.
        
        Args:
            cfg: Full configuration dict
            
        Returns:
            (trainloader, valloader) tuple
        """
        transform_cfg = cfg['transform_cfg']
        loader_cfg = cfg['dataloader_cfg']
        
        train_ds = get_dataset(cfg['dataset'], cfg['train_split'], root_dir=cfg['root_dir'], metadata=cfg['metadata'], bands=cfg['bands'])
        val_ds = get_dataset(cfg['dataset'], cfg['val_split'], root_dir=cfg['root_dir'], metadata=cfg['metadata'], bands=cfg['bands'])
        
        # Wrapper datasets
        trainset = SupervisedDataset(train_ds, transform_cfg['train'])
        valset = SupervisedDataset(val_ds, []) # Validation transforms handled by SegmentationInference

        trainloader = DataLoader(
            trainset, 
            batch_size=loader_cfg['batch_size'], 
            num_workers=loader_cfg['num_workers_train'], 
            pin_memory=loader_cfg['pin_memory'], 
            prefetch_factor=loader_cfg['prefetch_factor'], 
            persistent_workers=loader_cfg['persistent_workers'],
            drop_last=True,
            shuffle=True,
        )
        valloader = DataLoader(
            valset, 
            batch_size=loader_cfg['batch_size'], 
            num_workers=loader_cfg['num_workers_val'], 
            pin_memory=loader_cfg['pin_memory'], 
            prefetch_factor=loader_cfg['prefetch_factor'], 
            persistent_workers=loader_cfg['persistent_workers'],
            drop_last=False,
            shuffle=False,
        )

        return trainloader, valloader


if __name__ == '__main__':
    main()