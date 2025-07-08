import pytorch_lightning as pl
import pytorch


class NowcastingModel(pl.LightningModule):
    def __init__(self, learning_rate):
        super(NowcastingModel, self).__init__()
        self.learning_rate = learning_rate
        # make simple 1x1 convolution model
        self.model = pytorch.nn.Sequential(
            pytorch.nn.Conv2d(1, 1, kernel_size=1),
            pytorch.nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return loss
    
class NowcastingDataset(pytorch.Dataset):
    def __init__(self, path):
        self.data = load_data(path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    train_dataset = NowcastingDataset(path='train_data')
    train_dataloader = pytorch.data.DataLoader(train_dataset, batch_size=32)
    model = NowcastingModel(model)
    
    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader)

    eval_dataloader = ... # as above
    prediction = trainer.predict(model, eval_dataloader)
