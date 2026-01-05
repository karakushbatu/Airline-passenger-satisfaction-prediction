# ============================================================================
# MODEL MODÜLÜ - model.py
# ============================================================================
# Bu modül, makine öğrenmesi modellerini tanımlamak, eğitmek ve kaydetmek
# için gerekli sınıf ve fonksiyonları içerir. İki farklı sınıflandırıcı
# (Random Forest ve Logistic Regression) desteklenmektedir.
# ============================================================================

"""
Model Modülü

Bu modül, yolcu memnuniyeti tahminleme görevi için kullanılacak makine
öğrenmesi modellerini içerir. Scikit-learn tabanlı modeller kullanılarak
esnek ve genişletilebilir bir yapı sunulmaktadır.

Sınıflar:
    - SatisfactionModel: Ana model sınıfı, birden fazla algoritma destekler

Desteklenen Algoritmalar:
    - Random Forest Classifier
    - Logistic Regression
"""

import os
import joblib
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import warnings

warnings.filterwarnings('ignore')


class SatisfactionModel:
    """
    Yolcu memnuniyeti tahminleme modeli sınıfı.
    
    Bu sınıf, farklı makine öğrenmesi algoritmalarını tek bir arayüz
    altında toplar. Hem Random Forest hem de Logistic Regression
    algoritmalarını destekler.
    
    Özellikler:
    -----------
    - Birden fazla algoritma desteği
    - Model eğitimi ve tahmin
    - Model kaydetme ve yükleme (joblib)
    - Hiperparametre yapılandırması
    
    Kullanım:
    ---------
    >>> model = SatisfactionModel(model_type='random_forest')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> model.save('models/rf_model.pkl')
    
    Notlar:
    -------
    Varsayılan olarak Random Forest kullanılır çünkü genellikle
    tabular veri setlerinde iyi performans gösterir.
    """
    
    # Desteklenen model tipleri ve varsayılan hiperparametreleri
    MODEL_TYPES = {
        'random_forest': {
            'class': RandomForestClassifier,
            'default_params': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1  # Tüm CPU çekirdeklerini kullan
            }
        },
        'logistic_regression': {
            'class': LogisticRegression,
            'default_params': {
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'lbfgs',
                'n_jobs': -1
            }
        }
    }
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        custom_params: Optional[Dict[str, Any]] = None
    ):
        """
        SatisfactionModel sınıfının yapıcı metodu.
        
        Parametreler:
        ------------
        model_type : str, varsayılan='random_forest'
            Kullanılacak makine öğrenmesi algoritması.
            Seçenekler: 'random_forest', 'logistic_regression'
        
        custom_params : Dict[str, Any], opsiyonel
            Modelin hiperparametrelerini özelleştirmek için sözlük.
            Varsayılan parametrelerin üzerine yazılır.
        
        Hatalar:
        --------
        ValueError
            Geçersiz model tipi verildiğinde fırlatılır.
        
        Örnek:
        ------
        >>> # Varsayılan parametrelerle Random Forest
        >>> model_rf = SatisfactionModel('random_forest')
        
        >>> # Özel parametrelerle Logistic Regression
        >>> model_lr = SatisfactionModel(
        ...     model_type='logistic_regression',
        ...     custom_params={'C': 0.5, 'penalty': 'l2'}
        ... )
        """
        # Model tipini kontrol ediyoruz
        if model_type not in self.MODEL_TYPES:
            supported = list(self.MODEL_TYPES.keys())
            raise ValueError(
                f"Geçersiz model tipi: '{model_type}'. "
                f"Desteklenen tipler: {supported}"
            )
        
        self.model_type = model_type
        self.model_info = self.MODEL_TYPES[model_type]
        
        # Parametreleri birleştiriyoruz (varsayılan + özel)
        self.params = self.model_info['default_params'].copy()
        if custom_params:
            self.params.update(custom_params)
        
        # Model nesnesini oluşturuyoruz
        self.model: BaseEstimator = self.model_info['class'](**self.params)
        
        # Eğitim durumunu takip ediyoruz
        self._is_fitted: bool = False
        self._feature_count: Optional[int] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SatisfactionModel':
        """
        Modeli verilen eğitim verisi üzerinde eğitir.
        
        Bu metot, scikit-learn'ün fit() metodunu çağırarak modeli
        eğitir. Eğitim tamamlandığında model tahmin yapmaya hazır hale gelir.
        
        Parametreler:
        ------------
        X : np.ndarray
            Eğitim özellikleri (feature matrix).
            Boyut: (n_samples, n_features)
        
        y : np.ndarray
            Eğitim etiketleri (target array).
            Boyut: (n_samples,)
        
        Döndürür:
        ---------
        SatisfactionModel
            Eğitilmiş model nesnesi (zincirleme çağrı için self döndürülür).
        
        Örnek:
        ------
        >>> model = SatisfactionModel().fit(X_train, y_train)
        """
        self._feature_count = X.shape[1]
        self.model.fit(X, y)
        self._is_fitted = True
        
        print(f"  ✓ {self.get_model_name()} modeli başarıyla eğitildi")
        print(f"    - Eğitim örnek sayısı: {len(X):,}")
        print(f"    - Özellik sayısı: {self._feature_count}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Eğitilmiş model ile tahmin yapar.
        
        Parametreler:
        ------------
        X : np.ndarray
            Tahmin yapılacak özellik matrisi.
            Boyut: (n_samples, n_features)
        
        Döndürür:
        ---------
        np.ndarray
            Tahmin edilen sınıf etiketleri (0 veya 1).
        
        Hatalar:
        --------
        RuntimeError
            Model henüz eğitilmemişse fırlatılır.
        """
        self._check_is_fitted()
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Her sınıf için olasılık tahminleri döndürür.
        
        Bu metot, sadece sınıf etiketi yerine her sınıfın olasılığını
        döndürür. ROC-AUC gibi metriklerin hesaplanması için gereklidir.
        
        Parametreler:
        ------------
        X : np.ndarray
            Tahmin yapılacak özellik matrisi.
        
        Döndürür:
        ---------
        np.ndarray
            Olasılık matrisi. Boyut: (n_samples, 2)
            [:, 0] -> sınıf 0 olasılığı
            [:, 1] -> sınıf 1 olasılığı
        """
        self._check_is_fitted()
        return self.model.predict_proba(X)
    
    def _check_is_fitted(self) -> None:
        """
        Modelin eğitilmiş olup olmadığını kontrol eder.
        
        Eğer model henüz eğitilmemişse RuntimeError fırlatır.
        Bu, predict() ve predict_proba() metodları tarafından
        çağrılır.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.get_model_name()} modeli henüz eğitilmemiş! "
                f"Önce fit() metodunu çağırın."
            )
    
    def get_model_name(self) -> str:
        """
        Model tipinin kullanıcı dostu adını döndürür.
        
        Döndürür:
        ---------
        str
            Model adı (örn: 'Random Forest', 'Logistic Regression')
        """
        name_mapping = {
            'random_forest': 'Random Forest',
            'logistic_regression': 'Logistic Regression'
        }
        return name_mapping.get(self.model_type, self.model_type)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Modelin hiperparametrelerini döndürür.
        
        Döndürür:
        ---------
        Dict[str, Any]
            Model hiperparametreleri sözlüğü.
        """
        return self.params.copy()
    
    def save(self, filepath: str) -> None:
        """
        Eğitilmiş modeli dosyaya kaydeder.
        
        Model, joblib kütüphanesi kullanılarak pickle formatında
        kaydedilir. Bu format, büyük numpy array'leri verimli bir
        şekilde saklar.
        
        Parametreler:
        ------------
        filepath : str
            Modelin kaydedileceği dosya yolu.
            Örnek: 'models/satisfaction_model.pkl'
        
        Notlar:
        -------
        - Dosya uzantısı olarak .pkl veya .joblib önerilir
        - Klasör mevcut değilse otomatik oluşturulur
        
        Örnek:
        ------
        >>> model.save('models/random_forest_v1.pkl')
        """
        self._check_is_fitted()
        
        # Klasörün var olduğundan emin oluyoruz
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  - Klasör oluşturuldu: {directory}")
        
        # Model ve metadata'yı birlikte kaydediyoruz
        save_data = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_count': self._feature_count
        }
        
        joblib.dump(save_data, filepath)
        print(f"  ✓ Model kaydedildi: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SatisfactionModel':
        """
        Kaydedilmiş modeli dosyadan yükler.
        
        Bu class metodu, daha önce save() ile kaydedilmiş bir modeli
        yükler ve kullanıma hazır bir SatisfactionModel nesnesi döndürür.
        
        Parametreler:
        ------------
        filepath : str
            Modelin yükleneceği dosya yolu.
        
        Döndürür:
        ---------
        SatisfactionModel
            Yüklenmiş ve tahmine hazır model nesnesi.
        
        Hatalar:
        --------
        FileNotFoundError
            Dosya bulunamazsa fırlatılır.
        
        Örnek:
        ------
        >>> loaded_model = SatisfactionModel.load('models/satisfaction_model.pkl')
        >>> predictions = loaded_model.predict(X_new)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")
        
        # Kaydedilmiş veriyi yüklüyoruz
        save_data = joblib.load(filepath)
        
        # Yeni bir SatisfactionModel nesnesi oluşturuyoruz
        instance = cls(
            model_type=save_data['model_type'],
            custom_params=save_data['params']
        )
        
        # Eğitilmiş modeli ve metadata'yı geri yüklüyoruz
        instance.model = save_data['model']
        instance._is_fitted = True
        instance._feature_count = save_data['feature_count']
        
        print(f"  ✓ Model yüklendi: {filepath}")
        print(f"    - Model tipi: {instance.get_model_name()}")
        
        return instance
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Özellik önem skorlarını döndürür (sadece Random Forest için).
        
        Random Forest modeli için, her özelliğin karar sürecine
        ne kadar katkıda bulunduğunu gösteren önem skorlarını döndürür.
        
        Döndürür:
        ---------
        np.ndarray veya None
            Özellik önem skorları array'i. Logistic Regression için None.
        
        Notlar:
        -------
        Logistic Regression için bu metot None döndürür çünkü
        doğrudan özellik önemi hesaplanamaz (katsayılar farklı bir kavramdır).
        """
        self._check_is_fitted()
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            print("  ! Bu model tipi özellik önemi döndürmez.")
            return None


def create_models() -> List[SatisfactionModel]:
    """
    Her iki model tipinden birer örnek oluşturur.
    
    Bu fonksiyon, projedeki tüm modelleri karşılaştırmak için
    kullanılan modellerin listesini döndürür.
    
    Döndürür:
    ---------
    List[SatisfactionModel]
        [Random Forest, Logistic Regression] modelleri listesi.
    
    Örnek:
    ------
    >>> models = create_models()
    >>> for model in models:
    ...     model.fit(X_train, y_train)
    """
    models = [
        SatisfactionModel(model_type='random_forest'),
        SatisfactionModel(model_type='logistic_regression')
    ]
    
    print(f"✓ {len(models)} model oluşturuldu:")
    for model in models:
        print(f"  - {model.get_model_name()}")
    
    return models


# Bu modül doğrudan çalıştırıldığında bilgi verir
if __name__ == "__main__":
    print("Model Modülü - Bilgi")
    print("-" * 40)
    print("\nDesteklenen model tipleri:")
    for model_type in SatisfactionModel.MODEL_TYPES:
        print(f"  - {model_type}")
    print("\nKullanım için main.py dosyasını çalıştırın.")
