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

### Öğrendiklerim

Bu proje süresince makine öğrenmesi alanında birçok değerli deneyim kazandım:

1. **Uçtan Uca ML Pipeline Tasarımı**: Bir makine öğrenmesi projesinin sadece model eğitiminden ibaret olmadığını, veri ön işleme, feature engineering, model seçimi ve değerlendirme aşamalarının ne kadar kritik olduğunu öğrendim.

2. **Modüler Kod Yazımı**: Python'da `src/` klasörü altında modüller oluşturarak kodun tekrar kullanılabilirliğini ve okunabilirliğini artırmayı öğrendim. Bu yaklaşım, ilerideki projelerimde de uygulayacağım önemli bir yazılım mühendisliği pratiği oldu.

3. **Scikit-learn Ekosistemi**: StandardScaler, LabelEncoder, train_test_split gibi araçların doğru kullanımı ve model performans metriklerinin (accuracy, precision, recall, F1-score) nasıl yorumlanacağını kavradım.

4. **Random Forest vs Logistic Regression**: Farklı algoritmaların aynı veri seti üzerinde nasıl farklı performans gösterdiğini gözlemledim. Random Forest'ın non-linear ilişkileri yakalama konusunda Logistic Regression'a göre çok daha başarılı olduğunu deneyimledim.

### Karşılaştığım Zorluklar

1. **Eksik Değer Yönetimi**: Veri setindeki eksik değerlerin nasıl ele alınacağına karar vermek başlangıçta zorlandığım konulardan biriydi. Median imputation stratejisinin neden tercih edildiğini araştırarak bu sorunu aştım.

2. **Sınıf Dengesizliği**: Hedef değişkendeki sınıf dengesizliğinin (56.7% vs 43.3%) model performansını nasıl etkileyebileceğini anlamak ve stratified split kullanmanın önemini kavramak zaman aldı.

3. **Hiperparametre Ayarlaması**: Random Forest'ın `n_estimators`, `max_depth` gibi hiperparametrelerinin optimal değerlerini bulmak için deneme-yanılma yöntemini kullandım. İleride GridSearchCV veya RandomizedSearchCV kullanarak bu süreci otomatikleştirmeyi hedefliyorum.

### Geliştirme Önerileri

1. **Cross-Validation**: K-Fold cross-validation kullanarak modelin daha güvenilir bir şekilde değerlendirilmesi sağlanabilir.

2. **Feature Importance Analizi**: Random Forest'ın feature importance özelliği kullanılarak hangi özelliklerin memnuniyeti en çok etkilediği görselleştirilebilir.

3. **Gradient Boosting Modelleri**: XGBoost veya LightGBM gibi daha gelişmiş ensemble yöntemleri denenerek performans artırılabilir.

4. **Web API Entegrasyonu**: Flask veya FastAPI kullanılarak eğitilmiş modelin bir REST API olarak sunulması, gerçek dünya uygulamalarına entegrasyonu kolaylaştırabilir.

5. **Docker Konteynerizasyonu**: Projenin Docker konteynerı içinde paketlenmesi, farklı ortamlarda tutarlı çalışmasını sağlayabilir.

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
