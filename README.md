# Titanic Survival Classification — PyTorch

Binary sınıflandırma modeli: Titanic yolcularının hayatta kalıp kalmadığını tahmin eder.

## Proje Özeti

| | |
|---|---|
| **Veri Seti** | [Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) — 891 yolcu, 12 özellik |
| **Görev** | Binary Classification (Survived: 0 / 1) |
| **Model** | PyTorch Neural Network (3 katman) |
| **Test Accuracy** | ~%82 |

## Model Mimarisi

```
Input (8) → Linear(8→12) → Linear(12→12) → Linear(12→1) → Sigmoid
```

Loss: `BCEWithLogitsLoss` | Optimizer: `Adam (lr=0.01)`

## Preprocessing Adımları

1. Gereksiz sütunları at (`PassengerId`, `Name`, `Ticket`, `Cabin`)
2. Eksik değerleri doldur (`Age` → median, `Embarked` → mode)
3. Encoding (`Sex` → binary, `Embarked` → one-hot)
4. Feature engineering (`Fare` → log1p dönüşümü)
5. Scaling (`StandardScaler` — sayısal sütunlar)

## Kullanım

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn torchmetrics
```

```python
# Notebook'u çalıştır
jupyter notebook pytorch_titanic.ipynb
```

### Yeni Yolcu Tahmini

Notebook'un son hücresinde değerleri değiştir:

```python
pclass   = 3        # 1, 2, 3
sex      = 'male'   # 'male' / 'female'
age      = 25.0
sibsp    = 0
parch    = 0
fare     = 7.25
embarked = 'S'      # 'S', 'C', 'Q'
```

## Dosya Yapısı

```
├── pytorch_titanic.ipynb   # Ana notebook
├── data/
│   └── Titanic-Dataset.csv
└── README.md
```
