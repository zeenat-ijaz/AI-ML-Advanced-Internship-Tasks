"""
Task 3: Multimodal ML - Housing Price Prediction Using Images + Tabular Data

Dataset: Ahmed & Moustafa "Houses Dataset" (535 houses, 4 images each - bedroom, bathroom,
kitchen, frontal - plus tabular attributes: bedrooms, bathrooms, area, zipcode, price).
https://github.com/emanhamed/Houses-dataset

Approach:
  1. A small CNN extracts a feature vector from the house's frontal-view image.
  2. Tabular features (bedrooms, bathrooms, area, zipcode-derived avg price) are scaled.
  3. Image features + tabular features are concatenated ("late fusion") and fed through a
     small MLP regression head to predict price.
  4. Evaluated with MAE and RMSE.

Runs on CPU: small image size (64x64), shallow CNN, modest epoch count.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

DATA_DIR = "data/Houses-dataset-master/Houses Dataset"
INFO_FILE = os.path.join(DATA_DIR, "HousesInfo.txt")
IMG_SIZE = 64
EPOCHS = 25
BATCH_SIZE = 16
LR = 1e-3


def load_tabular():
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(INFO_FILE, sep=" ", header=None, names=cols)
    df["house_id"] = np.arange(1, len(df) + 1)
    return df


class HousesDataset(Dataset):
    def __init__(self, df, tabular_features, targets):
        self.df = df.reset_index(drop=True)
        self.tabular_features = tabular_features
        self.targets = targets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        house_id = self.df.iloc[idx]["house_id"]
        img_path = os.path.join(DATA_DIR, f"{house_id}_frontal.jpg")
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_arr).permute(2, 0, 1)
        tab_tensor = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img_tensor, tab_tensor, target


class MultimodalNet(nn.Module):
    def __init__(self, n_tabular_features):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
            nn.AdaptiveAvgPool2d(1),
        )
        self.image_fc = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU())

        self.tabular_fc = nn.Sequential(
            nn.Linear(n_tabular_features, 16), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(32 + 16, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, img, tab):
        img_feat = self.image_fc(self.cnn(img))
        tab_feat = self.tabular_fc(tab)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.head(fused).squeeze(1)


def main():
    df = load_tabular()
    print(f"Loaded {len(df)} houses.")

    # zipcode as a categorical one-hot-ish feature; simple approach for a small dataset
    df = pd.get_dummies(df, columns=["zipcode"], prefix="zip")

    feature_cols = [c for c in df.columns if c not in ["price", "house_id"]]
    X_tab = df[feature_cols].values.astype(np.float32)
    y = df["price"].values.astype(np.float32)

    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tab_train = scaler.fit_transform(X_tab[train_idx])
    X_tab_test = scaler.transform(X_tab[test_idx])

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y[train_idx].reshape(-1, 1)).flatten()
    y_test = y[test_idx]

    train_ds = HousesDataset(df.iloc[train_idx], X_tab_train, y_train)
    test_ds = HousesDataset(df.iloc[test_idx], X_tab_test, np.zeros(len(test_idx), dtype=np.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MultimodalNet(n_tabular_features=X_tab.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("Training multimodal model...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for img, tab, target in train_loader:
            optimizer.zero_grad()
            pred = model(img, tab)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * img.size(0)
        epoch_loss /= len(train_ds)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - train MSE (scaled): {epoch_loss:.4f}")

    print("Evaluating...")
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for img, tab, _ in test_loader:
            preds_scaled.append(model(img, tab).numpy())
    preds_scaled = np.concatenate(preds_scaled)
    preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"Test MAE: ${mae:,.2f}")
    print(f"Test RMSE: ${rmse:,.2f}")

    with open("eval_results.txt", "w") as f:
        f.write(f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\n")

    torch.save(model.state_dict(), "multimodal_model.pt")
    print("Saved model to multimodal_model.pt")


if __name__ == "__main__":
    main()
