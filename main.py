#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# ANA PIPELINE DOSYASI - main.py
# ============================================================================
# Bu dosya, Airline Passenger Satisfaction Prediction projesinin ana giriş
# noktasıdır. Tüm pipeline'ı orchestrate eder:
#   1. Veri Yükleme
#   2. Veri Ön İşleme
#   3. Model Eğitimi
#   4. Model Değerlendirmesi
#   5. En İyi Modelin Kaydedilmesi
# 
# Kullanım: python main.py
# ============================================================================

"""
Airline Passenger Satisfaction Prediction - Ana Pipeline

Bu script, uçtan uca makine öğrenmesi pipeline'ını çalıştırır.
Terminal'den doğrudan çalıştırılabilir.

TUSAŞ SKY Remote Staj Programı - Makine Öğrenmesi Projesi
"""

import os
import sys
import time
from datetime import datetime

# Proje kök dizinini Python path'ine ekliyoruz
# Bu sayede src modüllerini import edebiliriz
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Kendi modüllerimizi import ediyoruz
from src.data_loader import load_train_data
from src.preprocessing import DataPreprocessor
from src.model import SatisfactionModel, create_models
from src.train import train_models
from src.evaluate import (
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    compare_models
)


def print_banner():
    """
    Program başlangıcında hoş geldiniz banner'ı yazdırır.
    
    Bu fonksiyon, kullanıcıya programın ne yaptığını ve hangi
    versiyonda olduğunu gösterir.
    """
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          ✈️  AIRLINE PASSENGER SATISFACTION PREDICTION SYSTEM  ✈️           ║
║                                                                              ║
║                     TUSAŞ SKY Remote Staj Programı                          ║
║                      Makine Öğrenmesi Projesi                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"  📅 Çalıştırma Tarihi: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"  📂 Çalışma Dizini: {PROJECT_ROOT}")
    print()


def check_data_exists(data_path: str) -> bool:
    """
    Gerekli veri dosyalarının mevcut olup olmadığını kontrol eder.
    
    Parametreler:
    ------------
    data_path : str
        Veri dosyalarının bulunduğu klasör yolu.
    
    Döndürür:
    ---------
    bool
        Veri dosyaları mevcutsa True, değilse False.
    """
    train_file = os.path.join(data_path, "train.csv")
    
    if not os.path.exists(train_file):
        print("=" * 60)
        print("⚠️  VERİ SETİ BULUNAMADI!")
        print("=" * 60)
        print(f"\nBeklenen dosya yolu: {train_file}")
        print("\nLütfen aşağıdaki adımları izleyin:")
        print("1. Kaggle'dan veri setini indirin:")
        print("   https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction")
        print("2. İndirdiğiniz dosyaları 'data/raw/' klasörüne yerleştirin")
        print("3. Bu script'i tekrar çalıştırın: python main.py")
        print()
        return False
    
    return True


def run_pipeline(
    data_dir: str = "data/raw",
    model_save_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42,
    show_plots: bool = True
) -> dict:
    """
    Tüm makine öğrenmesi pipeline'ını çalıştırır.
    
    Bu fonksiyon, veri yüklemeden model değerlendirmesine kadar
    tüm adımları sırasıyla gerçekleştirir.
    
    Parametreler:
    ------------
    data_dir : str, varsayılan='data/raw'
        Ham veri dosyalarının bulunduğu klasör.
    
    model_save_dir : str, varsayılan='models'
        Eğitilmiş modellerin kaydedileceği klasör.
    
    test_size : float, varsayılan=0.2
        Test seti için ayrılacak veri oranı.
    
    random_state : int, varsayılan=42
        Tekrarlanabilirlik için rastgele tohum değeri.
    
    show_plots : bool, varsayılan=True
        Grafiklerin gösterilip gösterilmeyeceği.
    
    Döndürür:
    ---------
    dict
        Pipeline sonuçlarını içeren sözlük:
        - 'best_model': En iyi performans gösteren model
        - 'all_results': Tüm modellerin değerlendirme sonuçları
        - 'preprocessing_time': Ön işleme süresi
        - 'total_time': Toplam çalışma süresi
    """
    pipeline_start_time = time.time()
    results = {}
    
    # =========================================================================
    # ADIM 1: VERİ YÜKLEME
    # =========================================================================
    print("\n" + "=" * 60)
    print("📥 ADIM 1/5: VERİ YÜKLEME")
    print("=" * 60)
    
    step_start = time.time()
    df = load_train_data(data_dir)
    print(f"\n⏱  Veri yükleme süresi: {time.time() - step_start:.2f} saniye")
    
    # Veri hakkında kısa bilgi
    print(f"\n📊 Veri Seti Özeti:")
    print(f"  - Toplam örnek sayısı: {len(df):,}")
    print(f"  - Toplam özellik sayısı: {len(df.columns)}")
    print(f"  - Bellek kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # =========================================================================
    # ADIM 2: VERİ ÖN İŞLEME
    # =========================================================================
    print("\n" + "=" * 60)
    print("⚙️  ADIM 2/5: VERİ ÖN İŞLEME")
    print("=" * 60)
    
    step_start = time.time()
    preprocessor = DataPreprocessor(test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    preprocessing_time = time.time() - step_start
    
    results['preprocessing_time'] = preprocessing_time
    print(f"\n⏱  Ön işleme süresi: {preprocessing_time:.2f} saniye")
    
    # =========================================================================
    # ADIM 3: MODEL EĞİTİMİ
    # =========================================================================
    print("\n" + "=" * 60)
    print("🎓 ADIM 3/5: MODEL EĞİTİMİ")
    print("=" * 60)
    
    step_start = time.time()
    
    # Modelleri oluşturuyoruz
    models = create_models()
    
    # Modelleri eğitiyoruz (validation olarak test setini kullanıyoruz)
    training_results = train_models(
        models,
        X_train, y_train,
        X_test, y_test
    )
    
    training_time = time.time() - step_start
    print(f"\n⏱  Toplam eğitim süresi: {training_time:.2f} saniye")
    
    # =========================================================================
    # ADIM 4: MODEL DEĞERLENDİRMESİ
    # =========================================================================
    print("\n" + "=" * 60)
    print("📈 ADIM 4/5: MODEL DEĞERLENDİRMESİ")
    print("=" * 60)
    
    step_start = time.time()
    evaluation_results = []
    
    for result in training_results:
        model = result['model']
        eval_result = evaluate_model(model, X_test, y_test)
        evaluation_results.append(eval_result)
        
        # Detaylı sınıflandırma raporu
        print_classification_report(y_test, eval_result['predictions'])
    
    evaluation_time = time.time() - step_start
    print(f"\n⏱  Değerlendirme süresi: {evaluation_time:.2f} saniye")
    
    # Model karşılaştırması
    if len(evaluation_results) > 1:
        print("\n" + "=" * 60)
        print("📊 MODEL KARŞILAŞTIRMASI")
        print("=" * 60)
        
        if show_plots:
            try:
                compare_models(evaluation_results)
            except Exception as e:
                print(f"  ! Grafik gösterilirken hata: {e}")
    
    results['all_results'] = evaluation_results
    
    # =========================================================================
    # ADIM 5: EN İYİ MODELİ KAYDETME
    # =========================================================================
    print("\n" + "=" * 60)
    print("💾 ADIM 5/5: EN İYİ MODELİ KAYDETME")
    print("=" * 60)
    
    # En iyi modeli buluyoruz (en yüksek F1 skoruna göre)
    best_result = max(evaluation_results, key=lambda x: x['f1_score'])
    best_model_info = next(
        r for r in training_results 
        if r['name'] == best_result['model_name']
    )
    best_model = best_model_info['model']
    
    print(f"\n🏆 EN İYİ MODEL: {best_model.get_model_name()}")
    print(f"   F1-Skoru: {best_result['f1_score']:.4f}")
    print(f"   Doğruluk: {best_result['accuracy']:.4f}")
    
    # Modeli kaydediyoruz
    model_filename = "satisfaction_model.pkl"
    model_path = os.path.join(model_save_dir, model_filename)
    best_model.save(model_path)
    
    results['best_model'] = best_model
    
    # Karışıklık matrisini kaydediyoruz (her zaman PNG olarak kaydedilir)
    try:
        cm_path = os.path.join(model_save_dir, "confusion_matrix.png")
        # Matplotlib backend'i ayarla (GUI gerektirmeyen mod)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.evaluate import plot_confusion_matrix as plot_cm_func
        
        # Karışıklık matrisi grafiğini oluştur ve kaydet (show=False)
        cm = best_result['confusion_matrix']
        model_name = best_result.get('model_name', 'Model')
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Dissatisfied', 'Satisfied'],
            yticklabels=['Dissatisfied', 'Satisfied'],
            annot_kws={'size': 14},
            ax=ax
        )
        ax.set_title(f'Karışıklık Matrisi - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tahmin Edilen Sınıf', fontsize=12)
        ax.set_ylabel('Gerçek Sınıf', fontsize=12)
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Karışıklık matrisi kaydedildi: {cm_path}")
    except Exception as e:
        print(f"  ! Karışıklık matrisi kaydedilirken hata: {e}")
    
    # =========================================================================
    # SONUÇ ÖZETİ
    # =========================================================================
    total_time = time.time() - pipeline_start_time
    results['total_time'] = total_time
    
    print("\n" + "=" * 60)
    print("✅ PİPELINE BAŞARIYLA TAMAMLANDI!")
    print("=" * 60)
    print(f"\n📊 SONUÇ ÖZETİ:")
    print(f"  - En iyi model: {best_model.get_model_name()}")
    print(f"  - F1-Skoru: {best_result['f1_score']:.4f} ({best_result['f1_score']*100:.2f}%)")
    print(f"  - Doğruluk: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"  - Kesinlik: {best_result['precision']:.4f} ({best_result['precision']*100:.2f}%)")
    print(f"  - Duyarlılık: {best_result['recall']:.4f} ({best_result['recall']*100:.2f}%)")
    print(f"\n⏱  SÜRE BİLGİLERİ:")
    print(f"  - Ön işleme: {preprocessing_time:.2f} saniye")
    print(f"  - Eğitim: {training_time:.2f} saniye")
    print(f"  - Toplam: {total_time:.2f} saniye")
    print(f"\n💾 MODEL KAYDEDİLDİ:")
    print(f"  - {model_path}")
    print()
    
    return results


def main():
    """
    Ana fonksiyon - Script'in giriş noktası.
    
    Bu fonksiyon çağrıldığında:
    1. Banner'ı gösterir
    2. Veri dosyalarını kontrol eder
    3. Pipeline'ı çalıştırır
    """
    # Banner'ı gösteriyoruz
    print_banner()
    
    # Veri dizinini belirliyoruz
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    model_dir = os.path.join(PROJECT_ROOT, "models")
    
    # Veri dosyalarının varlığını kontrol ediyoruz
    if not check_data_exists(data_dir):
        sys.exit(1)
    
    try:
        # Pipeline'ı çalıştırıyoruz
        results = run_pipeline(
            data_dir=data_dir,
            model_save_dir=model_dir,
            test_size=0.2,
            random_state=42,
            show_plots=False  # Terminal'de çalışırken GUI bloklamasını önlemek için
        )
        
        print("=" * 60)
        print("  İyi çalışmalar! 🚀")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından iptal edildi.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# Script doğrudan çalıştırıldığında main() fonksiyonunu çağırıyoruz
if __name__ == "__main__":
    main()
