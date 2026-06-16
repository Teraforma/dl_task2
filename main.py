def main():

    import torchvision
    import numpy, sklearn
    import lightning as L
    from lightning_fabric.plugins.environments import lightning
    from models.alpha import Alpha
    from data.pets_dataset import get_dataloaders
    from models.classifier import PetClassifier


    # add test loader
    train_loader, val_loader = get_dataloaders()

    model = Alpha(num_classes=37)
    lightning_model = PetClassifier(model)

    trainer = L.Trainer(
        max_epochs=20,
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

if __name__ == "__main__":
    main()