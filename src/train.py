# ============================================================================
# EĞİTİM MODÜLÜ - train.py
# ============================================================================
# Bu modül, makine öğrenmesi modellerinin eğitilmesi için gerekli
# fonksiyonları içerir. Birden fazla modeli eğitip karşılaştırma imkanı sunar.
# ============================================================================

"""
Model Eğitim Modülü

Bu modül, SatisfactionModel sınıfını kullanarak modelleri eğitir
ve en iyi performans gösteren modeli belirler.

Fonksiyonlar:
    - train_model: Tek bir modeli eğitir
    - train_models: Birden fazla modeli eğitir ve karşılaştırır
    - train_and_select_best: Modelleri eğitir ve en iyisini seçer
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .model import SatisfactionModel, create_models
from sklearn.metrics import accuracy_score


def train_model(
    model: SatisfactionModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True
) -> Tuple[SatisfactionModel, float]:
    """
    Tek bir modeli eğitir ve eğitim süresini döndürür.
    
    Bu fonksiyon, verilen SatisfactionModel nesnesini eğitir ve
    eğitimin ne kadar sürdüğünü ölçer.
    
    Parametreler:
    ------------
    model : SatisfactionModel
        Eğitilecek model nesnesi.
    
    X_train : np.ndarray
        Eğitim özellikleri (feature matrix).
    
    y_train : np.ndarray
        Eğitim etiketleri (target array).
    
    verbose : bool, varsayılan=True
        True ise eğitim süreci hakkında bilgi yazdırılır.
    
    Döndürür:
    ---------
    Tuple[SatisfactionModel, float]
        (eğitilmiş_model, eğitim_süresi_saniye) tuple'ı.
    
    Örnek:
    ------
    >>> model = SatisfactionModel('random_forest')
    >>> trained_model, duration = train_model(model, X_train, y_train)
    >>> print(f"Eğitim süresi: {duration:.2f} saniye")
    """
    model_name = model.get_model_name()
    
    if verbose:
        print(f"\n[EĞİTİM] {model_name}")
        print("-" * 40)
    
    # Eğitim süresini ölçüyoruz
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    training_time = end_time - start_time
    
    if verbose:
        print(f"  ⏱  Eğitim süresi: {training_time:.2f} saniye")
    
    return model, training_time


def train_models(
    models: List[SatisfactionModel],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Birden fazla modeli eğitir ve sonuçları karşılaştırır.
    
    Bu fonksiyon, verilen model listesindeki tüm modelleri sırasıyla
    eğitir. Validasyon verisi sağlanmışsa, her model için doğruluk
    skoru hesaplanır.
    
    Parametreler:
    ------------
    models : List[SatisfactionModel]
        Eğitilecek model listesi.
    
    X_train : np.ndarray
        Eğitim özellikleri.
    
    y_train : np.ndarray
        Eğitim etiketleri.
    
    X_val : np.ndarray, opsiyonel
        Validasyon özellikleri (performans karşılaştırması için).
    
    y_val : np.ndarray, opsiyonel
        Validasyon etiketleri.
    
    Döndürür:
    ---------
    List[Dict[str, Any]]
        Her model için eğitim sonuçlarını içeren sözlük listesi.
        Her sözlük şunları içerir:
        - 'model': Eğitilmiş model nesnesi
        - 'name': Model adı
        - 'training_time': Eğitim süresi (saniye)
        - 'accuracy': Validasyon doğruluğu (varsa)
    
    Örnek:
    ------
    >>> models = create_models()
    >>> results = train_models(models, X_train, y_train, X_val, y_val)
    >>> for r in results:
    ...     print(f"{r['name']}: {r['accuracy']:.4f}")
    """
    print("\n" + "=" * 60)
    print("MODEL EĞİTİMİ BAŞLIYOR")
    print("=" * 60)
    print(f"Eğitilecek model sayısı: {len(models)}")
    print(f"Eğitim veri boyutu: {X_train.shape}")
    
    results = []
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model.get_model_name()} eğitiliyor...")
        
        # Modeli eğitiyoruz
        trained_model, training_time = train_model(model, X_train, y_train)
        
        # Sonuç sözlüğünü oluşturuyoruz
        result = {
            'model': trained_model,
            'name': trained_model.get_model_name(),
            'training_time': training_time,
            'accuracy': None
        }
        
        # Validasyon verisi varsa doğruluk hesaplıyoruz
        if X_val is not None and y_val is not None:
            y_pred = trained_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            result['accuracy'] = accuracy
            print(f"  🎯 Validasyon doğruluğu: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        results.append(result)
    
    print("\n" + "=" * 60)
    print("✓ TÜM MODELLERİN EĞİTİMİ TAMAMLANDI!")
    print("=" * 60)
    
    # Özet tablo
    print("\n📊 EĞİTİM ÖZETİ:")
    print("-" * 60)
    print(f"{'Model':<25} {'Süre (sn)':<15} {'Doğruluk':<15}")
    print("-" * 60)
    
    for r in results:
        acc_str = f"{r['accuracy']:.4f}" if r['accuracy'] else "N/A"
        print(f"{r['name']:<25} {r['training_time']:<15.2f} {acc_str:<15}")
    
    print("-" * 60)
    
    return results


def train_and_select_best(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_types: Optional[List[str]] = None
) -> Tuple[SatisfactionModel, List[Dict[str, Any]]]:
    """
    Modelleri eğitir ve en iyi performans göstereni seçer.
    
    Bu fonksiyon, belirtilen model tiplerini eğitir, validasyon seti
    üzerinde değerlendirir ve en yüksek doğruluk skoruna sahip modeli
    döndürür.
    
    Parametreler:
    ------------
    X_train : np.ndarray
        Eğitim özellikleri.
    
    y_train : np.ndarray
        Eğitim etiketleri.
    
    X_val : np.ndarray
        Validasyon özellikleri.
    
    y_val : np.ndarray
        Validasyon etiketleri.
    
    model_types : List[str], opsiyonel
        Eğitilecek model tipleri. Varsayılan: ['random_forest', 'logistic_regression']
    
    Döndürür:
    ---------
    Tuple[SatisfactionModel, List[Dict[str, Any]]]
        (en_iyi_model, tüm_sonuçlar) tuple'ı.
    
    Örnek:
    ------
    >>> best_model, results = train_and_select_best(
    ...     X_train, y_train, X_val, y_val
    ... )
    >>> print(f"En iyi model: {best_model.get_model_name()}")
    """
    # Varsayılan model tipleri
    if model_types is None:
        model_types = ['random_forest', 'logistic_regression']
    
    # Modelleri oluşturuyoruz
    models = [SatisfactionModel(model_type=mt) for mt in model_types]
    
    # Modelleri eğitiyoruz
    results = train_models(models, X_train, y_train, X_val, y_val)
    
    # En iyi modeli seçiyoruz (en yüksek doğruluk)
    best_result = max(results, key=lambda x: x['accuracy'] or 0)
    best_model = best_result['model']
    
    print(f"\n🏆 EN İYİ MODEL: {best_model.get_model_name()}")
    print(f"   Doğruluk: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    
    return best_model, results


# Bu modül doğrudan çalıştırıldığında bilgi verir
if __name__ == "__main__":
    print("Train Modülü - Bilgi")
    print("-" * 40)
    print("\nBu modül model eğitimi için kullanılır.")
    print("Kullanım için main.py dosyasını çalıştırın.")
