import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from datamodules.wsi_embedding_datamodule import PatchEmbeddingDataModule


class Trainer:

    def __init__(self, args, model, tokenizer, split_frac=(0.75, 0.12, 0.13)):
        self.ckpt_path = args.ckpt_path
        self.max_epochs = args.max_epochs
        self.split_frac = split_frac
        self.datamodule = PatchEmbeddingDataModule(args, tokenizer, split_frac)
        self.model = model
        pl.seed_everything(42)

    def train(self, fast_dev_run=False):
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.ckpt_path,  # Directory to save checkpoints
            filename="best_model",  # Naming convention
            monitor="val_loss",  # Metric to monitor for saving best checkpoints
            mode="min",  # Whether to minimize or maximize the monitored metric
            save_top_k=1,  # Number of best checkpoints to keep
            save_last=True  # Save the last checkpoint regardless of the monitored metric
        )
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=True, mode="min")
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            accelerator='gpu',
            devices=[0, 1, 2],
            strategy='ddp',
            enable_progress_bar=True,
            log_every_n_steps=2,
            fast_dev_run=fast_dev_run
        )
        train_metrics = trainer.fit(
            self.model, datamodule=self.datamodule
        )

        return train_metrics
