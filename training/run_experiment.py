
import importlib
import argparse
import albumentations
import pytorch_lightning as pl
import wandb


def _import_class(module_class_name):
    module_name, class_name = module_class_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)

    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = 'Trainer Args'
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    parser.add_argument('--data_class', type=str, default='CellClass')
    parser.add_argument('--model_class', type=str, default='Unet')
    parser.add_argument('--encoder_name', type=str, default='resnet34')
    parser.add_argument('--lit_model_class', type=str, default='BaseLitModel')
    parser.add_argument('--dir_out', type=str, default='training/logs')
    parser.add_argument('--wandb', action='store_true', default=False)

    args, _ = parser.parse_known_args()
    data_class = _import_class(f'seggit.data.{args.data_class}')
    lit_model_class = _import_class(f'seggit.lit_models.{args.lit_model_class}')
    data_class.add_argparse_args(parser)
    lit_model_class.add_argparse_args(parser)

    parser.add_argument('--help', '-h', action='help')
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    if args.data_class == 'InstanceDirection':
        assert args.model_class == 'DirectionNet'
        assert args.lit_model_class == 'InstanceDirectionLitModel'

    data_class = _import_class(f'seggit.data.{args.data_class}')
    model_class = _import_class(f'seggit.models.{args.model_class}')
    lit_model_class = _import_class(
        f'seggit.lit_models.{args.lit_model_class}')

    data = data_class(args)
    data.prepare_data()
    data.setup()

    if args.model_class == 'DirectionNet':
        model = model_class()
    else:
        model = model_class(encoder_name=args.encoder_name, 
                            in_channels=1,
                            classes=1, 
                            activation='sigmoid')

    lit_model = lit_model_class(model, args=args)

    logger = pl.loggers.TensorBoardLogger(args.dir_out)
    if args.wandb:
        if args.lit_model_class == 'InstanceDirectionLitModel':
            project = ('sartorius_cell_instance_segmentation_kaggle-'
                       'direction_net_training')
        else:
            project = 'sartorius_cell_instance_segmentation_kaggle-training'
        logger = pl.loggers.WandbLogger(project=project)
        logger.watch(model)
        logger.log_hyperparams(vars(args))

    if args.lit_model_class == 'InstanceDirectionLitModel':
        monitor = 'val_loss'
        mode = 'min'
    else:
        monitor = 'val_iou5'
        mode = 'max'
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=10)

    if args.lit_model_class == 'InstanceDirectionLitModel':
        filename = f'fold{args.fold:d}-' + '{epoch:03d}-{val_loss:.3f}'
        monitor = 'val_loss'
        mode = 'min'
    else:
        filename = (f'fold{args.fold:d}-' + 
                    '{epoch:03d}-{val_loss:.3f}-{val_iou5:.3f}')
        monitor = 'val_iou5'
        mode = 'max'
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=filename,
        monitor=monitor,
        mode=mode, 
        save_last=True)

    # callbacks = [early_stopping_callback,
    callbacks = [model_checkpoint_callback]

    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=logger,
                                            callbacks=callbacks,
                                            weights_summary='full',
                                            weights_save_path=args.dir_out)

    trainer.tune(model=lit_model, datamodule=data)
    trainer.fit(model=lit_model, datamodule=data)

    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        print(f'Best model saved at {best_model_path}.')
        if args.wandb:
            wandb.save(best_model_path)
            print('Best model also uploaded to W&B.')


if __name__ == '__main__':
    main()
