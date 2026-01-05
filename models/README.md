# Eğitilmiş Modeller

Bu klasör, eğitilmiş makine öğrenmesi modellerini içerir.

## Beklenen Dosyalar

- `satisfaction_model.pkl` - En iyi performans gösteren eğitilmiş model
- `confusion_matrix.png` - Karışıklık matrisi görselleştirmesi

## Model Yükleme

```python
from src.model import SatisfactionModel

# Modeli yükle
model = SatisfactionModel.load('models/satisfaction_model.pkl')

# Tahmin yap
predictions = model.predict(X_new)
```
