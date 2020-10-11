import pytorch_lightning as pl

from src.arguments import args
from src.utils import get_dataloaders
from src.Trainer.ae_base import AutoencoderTrainer
from src.vae.vanilla import VanillaVAE

num_epochs = args['num_epochs']
gpus = args['gpus']
dataset_path = args['dataset_path']
input_size = args['input_size']
batch_size = args['batch_size']

dataloaders = get_dataloaders(dataset_path, input_size, batch_size)

vae = VanillaVAE(z_dim=args['z_dim'])
ae_trainer = AutoencoderTrainer(vae)

trainer = pl.Trainer(max_epochs=num_epochs,
                     gpus=gpus)

trainer.fit(model=ae_trainer,
            train_dataloader=dataloaders['train'],
            val_dataloaders=dataloaders['val'])