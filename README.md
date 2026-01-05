# ✈️ Airline Passenger Satisfaction Prediction System

TUSAŞ SKY Remote Staj Programı kapsamında geliştirilen, uçak yolcularının memnuniyet durumunu tahmin eden makine öğrenmesi projesi.

---

## 📋 Proje Açıklaması

Bu proje, uçak yolcularının çeşitli hizmet değerlendirmelerine ve demografik bilgilerine dayanarak memnuniyet durumlarını (**satisfied** veya **neutral or dissatisfied**) tahmin eden bir sınıflandırma sistemidir.

### Proje Hedefleri
- Yolcu memnuniyetini etkileyen faktörlerin analizi
- Farklı makine öğrenmesi modellerinin karşılaştırılması
- Üretim ortamına hazır, modüler bir ML pipeline'ı oluşturma

---

## 📊 Veri Seti

Veri seti [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) platformundan temin edilmektedir.

### Veri Seti Özellikleri
| Özellik | Değer |
|---------|-------|
| Eğitim Örnekleri | ~103,000 |
| Test Örnekleri | ~26,000 |
| Özellik Sayısı | 22+ |
| Hedef Değişken | `satisfaction` (binary) |

### Öne Çıkan Özellikler
- **Demografik**: Cinsiyet, yaş, müşteri tipi
- **Seyahat**: Seyahat sınıfı, uçuş mesafesi, varış/kalkış gecikmesi
- **Hizmet Değerlendirmeleri**: Wi-Fi, online check-in, yemek, koltuk konforu vb.

---

## 🛠️ Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- pip paket yöneticisi

### Kurulum Adımları

1. **Depoyu klonlayın**
```bash
git clone https://github.com/your-username/airline-passenger-satisfaction.git
cd airline-passenger-satisfaction
```

2. **Sanal ortam oluşturun (önerilen)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows
```

3. **Gereksinimleri yükleyin**
```bash
pip install -r requirements.txt
```

4. **Veri setini indirin**
   - [Kaggle sayfasından](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) veri setini indirin
   - `train.csv` ve `test.csv` dosyalarını `data/raw/` klasörüne yerleştirin

---

## 🚀 Çalıştırma

### Ana Pipeline'ı Çalıştırma
```bash
python main.py
```

Bu komut aşağıdaki adımları sırasıyla gerçekleştirir:
1. ✅ Veri yükleme
2. ✅ Veri ön işleme (eksik değer doldurma, encoding, scaling)
3. ✅ Model eğitimi (Random Forest + Logistic Regression)
4. ✅ Model değerlendirmesi
5. ✅ En iyi modelin kaydedilmesi

### Jupyter Notebook ile Keşifsel Veri Analizi (EDA)
```bash
jupyter notebook notebooks/eda.ipynb
```

---

## 📁 Proje Yapısı

```text
airline-passenger-satisfaction/
├── data/
│   ├── raw/                    # Ham veri dosyaları (train.csv, test.csv)
│   └── processed/              # İşlenmiş veri dosyaları
├── notebooks/
│   └── eda.ipynb               # Keşifsel veri analizi notebook'u
├── src/
│   ├── __init__.py             # Paket başlatma dosyası
│   ├── data_loader.py          # Veri yükleme fonksiyonları
│   ├── preprocessing.py        # Veri ön işleme pipeline'ı
│   ├── model.py                # Model sınıfları (RF, LR)
│   ├── train.py                # Model eğitim fonksiyonları
│   └── evaluate.py             # Model değerlendirme metrikleri
├── models/
│   └── satisfaction_model.pkl  # Eğitilmiş model dosyası
├── main.py                     # Ana pipeline script'i
├── requirements.txt            # Python bağımlılıkları
└── README.md                   # Bu dosya
```

---

## 🔬 Kullanılan Teknolojiler

| Kategori | Teknoloji |
|----------|-----------|
| Programlama Dili | Python 3.8+ |
| Veri İşleme | Pandas, NumPy |
| Makine Öğrenmesi | Scikit-learn |
| Görselleştirme | Matplotlib, Seaborn |
| Model Kaydetme | Joblib |
| Notebook | Jupyter |

---

## 📈 Sonuçlar

### Model Performansları

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **95.86%** | **96.13%** | **94.25%** | **95.18%** | **99.33%** |
| Logistic Regression | 87.66% | 87.00% | 84.08% | 85.51% | 92.78% |

### 🏆 En İyi Model: Random Forest

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **F1-Skoru** | 95.18% | Precision ve Recall'ın dengeli ortalaması |
| **Doğruluk** | 95.86% | 100 yolcudan 96'sının memnuniyetini doğru tahmin ediyor |
| **ROC-AUC** | 99.33% | Modelin sınıfları ayırt etme yeteneği mükemmel seviyede |

### Confusion Matrix (Random Forest)

|  | Tahmin: Dissatisfied | Tahmin: Satisfied |
|--|---------------------|-------------------|
| **Gerçek: Dissatisfied** | 11,434 (TN) | 342 (FP) |
| **Gerçek: Satisfied** | 518 (FN) | 8,487 (TP) |

### 📊 Teknik Yorumlar

1. **Model Karşılaştırması**: Random Forest, Logistic Regression'a göre ~%8 daha yüksek doğruluk sağlıyor. Bu fark, veri setindeki non-linear ilişkileri Random Forest'ın daha iyi yakalayabilmesinden kaynaklanıyor.

2. **Sınıf Dengesi**: Veri setinde %56.7 Dissatisfied, %43.3 Satisfied oranı var. Bu nispeten dengeli bir dağılım olup, model performansını olumsuz etkilemiyor.

3. **Yanlış Tahminler**: 
   - 342 yolcu yanlışlıkla "memnun" olarak sınıflandırıldı (False Positive)
   - 518 yolcu yanlışlıkla "memnun değil" olarak sınıflandırıldı (False Negative)
   - Toplam hata oranı sadece %4.14

4. **Özellik Mühendisliği**: Hizmet değerlendirmeleri (Online boarding, Inflight wifi service, Seat comfort) memnuniyetle en yüksek korelasyonu gösteriyor.

---

## 📝 Kişisel Değerlendirme

> Bu bölüm stajyer tarafından doldurulacaktır.

### Öğrendiklerim
<!-- Proje sürecinde öğrendiğiniz teknik ve kavramsal bilgileri buraya yazın -->

### Karşılaştığım Zorluklar
<!-- Proje sırasında karşılaştığınız zorlukları ve çözümlerini yazın -->

### Geliştirme Önerileri
<!-- Projeyi nasıl daha da geliştirebileceğinize dair fikirlerinizi yazın -->

---

## 🤝 Katkıda Bulunma

Bu proje TUSAŞ SKY Remote staj programı kapsamında geliştirilmiştir. Katkılarınız için pull request gönderebilirsiniz.

---

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

---

## 📞 İletişim

**Geliştirici:** [Adınız]  
**E-posta:** [E-posta adresiniz]  
**LinkedIn:** [LinkedIn profiliniz]

---

*Bu proje TUSAŞ SKY Remote Staj Programı için hazırlanmıştır.*
