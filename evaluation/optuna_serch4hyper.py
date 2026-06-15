from models.alpha import Alpha
from models.classifier import PetClassifier
import lightning as L

def objective(trial):

    lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-2,
        log=True
    )

    model = Alpha(
        dropout_rate=trial.suggest_float(
            "dropout",
            0.0,
            0.5
        )
    )

    lightning_model = PetClassifier(
        model,
        lr
    )

    trainer = L.Trainer(
        max_epochs=10
    )

    trainer.fit(...)

    return trainer.callback_metrics[
        "val_acc"
    ].item()