# Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data

## Objective
Predict housing prices using both structured (tabular) data and house images.

## Dataset
[Houses Dataset (Ahmed & Moustafa, 2016)](https://github.com/emanhamed/Houses-dataset) — 535
houses, each with 4 images (bedroom, bathroom, kitchen, frontal view) and tabular attributes
(bedrooms, bathrooms, area, zip code, price). We use the **frontal-view image** per house
alongside its tabular features.

## Methodology / Approach
1. **Tabular preprocessing**: one-hot encode zip code, standard-scale numeric features
   (bedrooms, bathrooms, area) and the target (price) for stable training.
2. **Image pipeline**: load each house's frontal image, resize to 64x64, normalize to [0, 1].
3. **CNN feature extraction**: a small 3-block convolutional network (Conv-ReLU-MaxPool x3 +
   global average pooling) extracts a 64-d embedding from each image, projected to 32-d.
4. **Feature fusion**: the 32-d image embedding is concatenated with a 16-d tabular embedding
   (from a small dense layer) — late/mid-level fusion — then passed through an MLP regression
   head to predict price.
5. Trained end-to-end (CNN + tabular branch + fusion head jointly) with MSE loss and Adam.
6. Evaluated on a held-out 20% split using **MAE** and **RMSE** (in original dollar scale, via
   inverse-transforming the scaled predictions).

## How to Run

```bash
pip install -r requirements.txt

# Downloads dataset first if not already present, see data/ setup below
python train_multimodal.py
```

Dataset setup: download and unzip
[emanhamed/Houses-dataset](https://github.com/emanhamed/Houses-dataset) into
`data/Houses-dataset-master/`.

## Results
See `eval_results.txt` (generated after training) for test MAE and RMSE.

## Key Results / Insights
- Fusing a compact CNN image embedding with tabular features outperforms tabular-only baselines
  on this dataset in prior literature, since visual cues (property condition, finish quality)
  carry price-relevant signal not captured by bedrooms/bathrooms/area/zip alone.
- With only ~535 houses, the CNN branch is intentionally shallow (3 conv blocks) to avoid
  overfitting; a pretrained backbone (e.g. ResNet features, frozen) would likely improve results
  further with more data or fine-tuning.
- Scaling the target (price) before training stabilized MSE-based training given the wide price
  range in the dataset.

## Skills Gained
- Multimodal machine learning
- Convolutional Neural Networks (CNNs)
- Feature fusion (image + tabular)
- Regression modeling and evaluation (MAE, RMSE)
