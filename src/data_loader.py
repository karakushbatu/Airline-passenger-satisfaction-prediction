# ============================================================================
# VERİ YÜKLEME MODÜLÜ - data_loader.py
# ============================================================================
# Bu modül, CSV formatındaki veri setlerini okumak için gerekli fonksiyonları
# içerir. Dosya bulunamadığında kullanıcıya anlaşılır hata mesajları verir.
# ============================================================================

"""
Veri Yükleme Modülü

Bu modül, Kaggle'dan indirilen airline passenger satisfaction veri setini
yüklemek için kullanılır. Hem eğitim (train) hem de test veri setlerini
destekler.

Fonksiyonlar:
    - load_data: Belirtilen CSV dosyasını pandas DataFrame olarak yükler
    - load_train_data: Eğitim veri setini yükler
    - load_test_data: Test veri setini yükler
"""

import os
import pandas as pd
from typing import Optional, Tuple


def load_data(file_path: str) -> pd.DataFrame:
    """
    Belirtilen dosya yolundan veri setini okur ve pandas DataFrame olarak döndürür.
    
    Bu fonksiyon, CSV formatındaki veri dosyalarını okumak için kullanılır.
    Dosya bulunamadığında veya okunamadığında anlaşılır hata mesajları verir.
    
    Parametreler:
    ------------
    file_path : str
        Okunacak CSV dosyasının tam yolu. Örnek: 'data/raw/train.csv'
    
    Döndürür:
    ---------
    pd.DataFrame
        Yüklenen veri setini içeren pandas DataFrame nesnesi.
    
    Hatalar:
    --------
    FileNotFoundError
        Belirtilen dosya yolunda dosya bulunamazsa fırlatılır.
    pd.errors.EmptyDataError
        Dosya boşsa fırlatılır.
    
    Kullanım Örneği:
    ----------------
    >>> from src.data_loader import load_data
    >>> df = load_data('data/raw/train.csv')
    >>> print(df.head())
    """
    # Dosyanın var olup olmadığını kontrol ediyoruz
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Hata: '{file_path}' dosyası bulunamadı. "
            f"Lütfen dosya yolunu kontrol edin veya veri setini "
            f"Kaggle'dan indirip 'data/raw/' klasörüne yerleştirin."
        )
    
    try:
        # CSV dosyasını okuyoruz
        # low_memory=False parametresi büyük dosyalarda veri tipi uyarılarını önler
        df = pd.read_csv(file_path, low_memory=False)
        
        # Başarılı yükleme mesajı
        satir_sayisi = len(df)
        sutun_sayisi = len(df.columns)
        print(f"✓ Veri seti başarıyla yüklendi: {satir_sayisi:,} satır, {sutun_sayisi} sütun")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(
            f"Hata: '{file_path}' dosyası boş. "
            f"Lütfen dosyanın içeriğini kontrol edin."
        )
    except Exception as e:
        raise Exception(
            f"Veri yüklenirken beklenmeyen bir hata oluştu: {str(e)}"
        )


def load_train_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Eğitim (train) veri setini yükler.
    
    Kaggle veri setindeki 'train.csv' dosyasını okur. Bu veri seti,
    modelin eğitilmesi için kullanılacak etiketli örnekleri içerir.
    
    Parametreler:
    ------------
    data_dir : str, varsayılan='data/raw'
        Veri dosyalarının bulunduğu klasör yolu.
    
    Döndürür:
    ---------
    pd.DataFrame
        Eğitim veri setini içeren DataFrame.
    """
    train_path = os.path.join(data_dir, "train.csv")
    print("Eğitim veri seti yükleniyor...")
    return load_data(train_path)


def load_test_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Test veri setini yükler.
    
    Kaggle veri setindeki 'test.csv' dosyasını okur. Bu veri seti,
    modelin performansını değerlendirmek için kullanılır.
    
    Parametreler:
    ------------
    data_dir : str, varsayılan='data/raw'
        Veri dosyalarının bulunduğu klasör yolu.
    
    Döndürür:
    ---------
    pd.DataFrame
        Test veri setini içeren DataFrame.
    """
    test_path = os.path.join(data_dir, "test.csv")
    print("Test veri seti yükleniyor...")
    return load_data(test_path)


def load_all_data(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hem eğitim hem de test veri setlerini tek seferde yükler.
    
    Bu fonksiyon, her iki veri setini de yükleyip tuple olarak döndürür.
    Pipeline'ın başlangıcında tüm verileri yüklemek için kullanışlıdır.
    
    Parametreler:
    ------------
    data_dir : str, varsayılan='data/raw'
        Veri dosyalarının bulunduğu klasör yolu.
    
    Döndürür:
    ---------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) şeklinde iki DataFrame içeren tuple.
    
    Kullanım Örneği:
    ----------------
    >>> train_df, test_df = load_all_data()
    >>> print(f"Eğitim: {len(train_df)} satır, Test: {len(test_df)} satır")
    """
    print("=" * 60)
    print("TÜM VERİ SETLERİ YÜKLENİYOR")
    print("=" * 60)
    
    train_df = load_train_data(data_dir)
    test_df = load_test_data(data_dir)
    
    print("=" * 60)
    print("✓ Tüm veri setleri başarıyla yüklendi!")
    print("=" * 60)
    
    return train_df, test_df


# Bu modül doğrudan çalıştırıldığında test amaçlı örnek kullanım
if __name__ == "__main__":
    # Modülü test etmek için örnek kullanım
    print("Data Loader Modülü - Test")
    print("-" * 40)
    
    try:
        # Eğitim verisini yüklemeyi dene
        df = load_train_data()
        print(f"\nVeri seti boyutu: {df.shape}")
        print(f"\nİlk 5 satır:")
        print(df.head())
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nLütfen veri setini Kaggle'dan indirin:")
        print("https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction")
