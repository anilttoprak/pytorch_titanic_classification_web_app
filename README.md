# Titanic Survival Classification — PyTorch

PyTorch ile binary sınıflandırma: Titanic yolcularının hayatta kalıp kalmadığını tahmin eder.

## Sonuçlar

| Metrik | Değer |
|---|---|
| Train Accuracy | ~%85 |
| Test Accuracy | ~%83 |
| Loss Fonksiyonu | BCEWithLogitsLoss |
| Optimizer | Adam (lr=0.01) |
| Epoch | 100 |

## Model Mimarisi

```
Input(8) → Linear(8→20) → ReLU → Linear(20→20) → ReLU → Linear(20→1) → Sigmoid
```

## Özellikler (8 Feature)

| Feature | Açıklama | İşlem |
|---|---|---|
| Pclass | Yolcu sınıfı (1/2/3) | — |
| Sex | Cinsiyet | Binary encoding (male=0, female=1) |
| Age | Yaş | Median ile dolduruldu → StandardScaler |
| SibSp | Kardeş/eş sayısı | StandardScaler |
| Parch | Ebeveyn/çocuk sayısı | StandardScaler |
| Fare | Bilet ücreti | log1p → StandardScaler |
| Embarked_Q | Queenstown'dan bindi | One-hot encoding |
| Embarked_S | Southampton'dan bindi | One-hot encoding |

## Notebook İçeriği

```
1. EDA
   ├── Eksik değer analizi
   ├── Survived dağılımı
   ├── Cinsiyet / Sınıf / Yaş / Ücret grafikleri
   ├── Korelasyon matrisi
   └── Pairplot

2. Preprocessing
   ├── 2.1 Gereksiz sütunları at (PassengerId, Name, Ticket, Cabin)
   ├── 2.2 Eksik değerleri doldur (Age→median, Embarked→mode)
   ├── 2.3 Outlier analizi (IQR + boxplot)
   ├── 2.4 Encoding (Sex→binary, Embarked→one-hot)
   ├── 2.5 Feature engineering (Fare→log1p)
   └── 2.6 Scaling (StandardScaler)

3. Model Eğitimi (100 epoch)
4. Confusion Matrix
5. Feature Importance (Permutation)
6. Yeni Yolcu Tahmini
```

## Kurulum

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn torchmetrics
```

## Kullanım

```bash
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
fare     = 7.25     # £ cinsinden
embarked = 'S'      # 'S'=Southampton, 'C'=Cherbourg, 'Q'=Queenstown
```

## Dosya Yapısı

```
├── pytorch_titanic.ipynb   # Ana notebook
├── models/
│   └── titanic_model.pth   # Kaydedilmiş model ağırlıkları
├── scaler.pkl              # Kaydedilmiş StandardScaler
├── data/
│   └── Titanic-Dataset.csv
└── README.md
```

