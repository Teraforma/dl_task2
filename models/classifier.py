import lightning as L
import torch
import torch.nn.functional as F

class PetClassifier(L.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = F.cross_entropy(out, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        out = self.model(x)
        loss = F.cross_entropy(out, y)

        acc =( # calculates accuracy
            out.argmax(dim=1) == y # takes biggest number for each
            # image and compares this prediction with the label
        ).float().mean() # bool to float, then mean value out of it

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)