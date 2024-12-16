from pathlib import Path

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import warnings
warnings.filterwarnings('ignore')

seed_everything(0)
torch.set_float32_matmul_precision('medium')

from model import ActionDetectionModel
from dataloaders.train_dataloader import train_dataloader
from dataloaders.test_dataloader import test_dataloader

dataset_path = Path(r'C:\Users\zedon\Videos\PW2024VIDEOS\clips\output')

model = ActionDetectionModel(
    learning_rate = 1e-3,
    batch_size = 6,
    num_worker = 0
)

checkpoint_callback = ModelCheckpoint(
    monitor = 'val_loss',
    dirpath = 'checkpoints',
    filename = 'model-{epoch:02d}-{val_loss:.2f}',
    save_last=True
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = Trainer(
    max_epochs = 200,
    accelerator = 'gpu',
    devices = -1, #-1
    precision = '16-mixed', # 16
    accumulate_grad_batches = 2,
    enable_progress_bar = True,
    num_sanity_val_steps = 0,
    callbacks = [lr_monitor, checkpoint_callback],
)

if __name__ == '__main__':
    train_loader = train_dataloader(dataset_path / 'train', batch_size=16, num_workers=0)
    val_loader = test_dataloader(dataset_path / 'val', batch_size=16, num_workers=0)
    test_loader = test_dataloader(dataset_path / 'test', batch_size=16, num_workers=0)
    ckpt_path = r'app\ml_models\action_detection\weights\model-v1.pth'
    # trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path) #ckpt_path='checkpoints/last.ckpt'
    # trainer.validate(model)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    trainer.test(model, test_loader)

    # torch.save(model.state_dict(), 'model-v1.pth')