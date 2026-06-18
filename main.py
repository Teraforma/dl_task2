def train():

    import torchvision
    import numpy, sklearn
    import lightning as L
    from lightning_fabric.plugins.environments import lightning
    from models.alpha import Alpha
    from data.pets_dataset import get_dataloaders
    from models.classifier import PetClassifier

    from pytorch_lightning.callbacks import ModelCheckpoint
    # add test loader
    train_loader, val_loader = get_dataloaders()

    model = Alpha(num_classes=37)
    lightning_model = PetClassifier(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="pet-model--{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=20,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # print("Best model:", checkpoint_callback.best_model_path)
def searchBest():
    # print("Best model:", checkpoint_callback.best_model_path)
    # %%
    import optuna
    from evaluation.optuna_serch4hyper import objective

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=20)

    print(study.best_params)


if __name__ == "__main__":
    train()

    searchBest()