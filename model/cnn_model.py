import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, num_classes):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)


    def forward(self, x):
        x = x.unsqueeze(1)  # Добавляем размер для канала
        x = x.permute(0, 2, 1)
        x1 = torch.relu(self.conv1(x))
        x1 = torch.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = torch.relu(self.conv2(x))
        x2 = torch.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = torch.relu(self.conv3(x))
        x3 = torch.max_pool1d(x3, x3.size(2)).squeeze(2)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        return self.fc(x)


def train_model(model, train_loader, val_loader=None, num_epochs=10):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Валидация
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")
