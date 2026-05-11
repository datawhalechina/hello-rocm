# file: src/infra/embrace-amd-ai/code/resnet_cifar10_amd.py
import random
import datetime
import torch
import torchvision
from datasets import load_dataset
import matplotlib.pyplot as plt


def get_dataloaders(batch_size=256):
    dataset = load_dataset("cifar10")
    dataset.set_format("torch")

    train_loader = torch.utils.data.DataLoader(
        dataset["train"], shuffle=True, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        dataset["test"], batch_size=batch_size
    )
    return train_loader, test_loader


def get_transform():
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

    def transform(x):
        if x.ndim == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        x = (x - mean.to(x.device)) / std.to(x.device)
        return x

    return transform


def build_model():
    model = torchvision.models.resnet18(num_classes=10)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    return model, loss_fn, optimizer


def train_model(model, loss_fn, optimizer, train_loader, test_loader, transform, num_epochs):
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    accuracy = []
    t0 = datetime.datetime.now()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        t0_epoch_train = datetime.datetime.now()

        model.train()
        train_losses, n_examples = [], 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            preds = model(transform(batch["img"]))
            loss = loss_fn(preds, batch["label"])
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach())
            n_examples += batch["label"].shape[0]

        train_loss = torch.stack(train_losses).mean().item()
        t_epoch_train = datetime.datetime.now() - t0_epoch_train

        model.eval()
        with torch.no_grad():
            t0_epoch_test = datetime.datetime.now()
            test_losses, n_test_examples, n_test_correct = [], 0, 0
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                preds = model(transform(batch["img"]))
                loss = loss_fn(preds, batch["label"])

                test_losses.append(loss)
                n_test_examples += batch["img"].shape[0]
                n_test_correct += (batch["label"] == preds.argmax(dim=1)).sum()

            test_loss = torch.stack(test_losses).mean().item()
            test_accuracy = n_test_correct / n_test_examples
            t_epoch_test = datetime.datetime.now() - t0_epoch_test
            accuracy.append(test_accuracy.cpu())

        print(f"  Epoch time: {t_epoch_train+t_epoch_test}")
        print(f"  Examples/second (train): {n_examples/t_epoch_train.total_seconds():0.4g}")
        print(f"  Examples/second (test): {n_test_examples/t_epoch_test.total_seconds():0.4g}")
        print(f"  Train loss: {train_loss:0.4g}")
        print(f"  Test loss: {test_loss:0.4g}")
        print(f"  Test accuracy: {test_accuracy*100:0.4g}%")

    total_time = datetime.datetime.now() - t0
    print(f"Total training time: {total_time}")
    return accuracy


def main():
    torch.manual_seed(0)
    random.seed(0)

    model, loss, optimizer = build_model()
    train_loader, test_loader = get_dataloaders()
    transform = get_transform()

    test_accuracy = train_model(
        model, loss, optimizer, train_loader, test_loader, transform, num_epochs=8
    )

    plt.plot(test_accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("ResNet18 on CIFAR10 (AMD ROCm)")
    plt.savefig("resnet_cifar10_amd.png")
    print("训练完成，准确率曲线已保存为 resnet_cifar10_amd.png")


if __name__ == "__main__":
    main()
