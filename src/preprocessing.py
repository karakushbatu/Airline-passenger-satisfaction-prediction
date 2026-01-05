# ============================================================================
# VERİ ÖN İŞLEME MODÜLÜ - preprocessing.py
# ============================================================================
# Bu modül, makine öğrenmesi için veri hazırlığı işlemlerini içerir:
# - Eksik değerlerin doldurulması (imputation)
# - Kategorik değişkenlerin sayısallaştırılması (encoding)
# - Özelliklerin ölçeklendirilmesi (scaling)
# - Veri setinin eğitim/test olarak bölünmesi (splitting)
# ============================================================================

"""
Veri Ön İşleme Modülü

Bu modül, ham veri setini makine öğrenmesi modellerine uygun hale getirmek
için gerekli tüm dönüşümleri sağlar. Scikit-learn'ün transformer API'si ile
uyumlu bir şekilde tasarlanmıştır.

Sınıflar:
    - DataPreprocessor: Tüm ön işleme adımlarını yöneten ana sınıf

Fonksiyonlar:
    - preprocess_data: Hızlı kullanım için wrapper fonksiyon
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings

# Uyarıları bastırıyoruz (ConvergenceWarning vb.)
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Veri ön işleme pipeline'ı için ana sınıf.
    
    Bu sınıf, ham veri setini makine öğrenmesi modelleri için hazırlamak üzere
    gerekli tüm dönüşümleri sırasıyla uygular. Fit-transform paradigmasını
    kullanarak eğitim ve test verilerini tutarlı bir şekilde işler.
    
    Özellikler:
    -----------
    - Eksik değerlerin akıllı doldurulması (sayısal: medyan, kategorik: mod)
    - Kategorik değişkenlerin Label Encoding ile sayısallaştırılması
    - Özelliklerin StandardScaler ile normalleştirilmesi
    - Hedef değişkenin (satisfaction) binary encode edilmesi
    
    Kullanım:
    ---------
    >>> preprocessor = DataPreprocessor()
    >>> X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    Notlar:
    -------
    - fit_transform() eğitim verisi için kullanılır
    - transform() test verisi için kullanılır (önceden fit edilmiş olmalı)
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        DataPreprocessor sınıfının yapıcı metodu.
        
        Parametreler:
        ------------
        test_size : float, varsayılan=0.2
            Test seti için ayrılacak veri oranı (0.0 - 1.0 arası).
            0.2 değeri verinin %20'sinin test için kullanılacağı anlamına gelir.
        
        random_state : int, varsayılan=42
            Rastgele sayı üreteci için tohum değeri.
            Aynı değer kullanıldığında her çalıştırmada aynı bölünme elde edilir.
        """
        self.test_size = test_size
        self.random_state = random_state
        
        # Encoder ve scaler nesnelerini saklayacağız (test verisi için gerekli)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.numeric_imputer: Optional[SimpleImputer] = None
        self.categorical_imputer: Optional[SimpleImputer] = None
        
        # Sütun bilgilerini saklıyoruz
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.target_column: str = "satisfaction"
        self.columns_to_drop: List[str] = ["Unnamed: 0", "id"]
        
        # Fit durumunu takip ediyoruz
        self._is_fitted: bool = False
    
    def _identify_column_types(self, df: pd.DataFrame) -> None:
        """
        DataFrame'deki sütunları sayısal ve kategorik olarak sınıflandırır.
        
        Bu metot, veri setindeki her sütunun tipini otomatik olarak belirler
        ve ilgili listelere ekler. Hedef değişken ve atılacak sütunlar hariç tutulur.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            Sütun tiplerinin belirleneceği DataFrame.
        """
        # Atılacak ve hedef sütunları hariç tutuyoruz
        columns_to_exclude = self.columns_to_drop + [self.target_column]
        
        for column in df.columns:
            if column in columns_to_exclude:
                continue
            
            # Sütun tipine göre sınıflandırma yapıyoruz
            if df[column].dtype in ['int64', 'float64']:
                self.numeric_columns.append(column)
            else:
                self.categorical_columns.append(column)
        
        print(f"  - Sayısal sütunlar ({len(self.numeric_columns)}): {self.numeric_columns[:5]}...")
        print(f"  - Kategorik sütunlar ({len(self.categorical_columns)}): {self.categorical_columns}")
    
    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Eksik değerleri (NaN) uygun yöntemlerle doldurur.
        
        Sayısal sütunlar için medyan değeri, kategorik sütunlar için ise
        en sık görülen değer (mod) kullanılır. Bu yaklaşım, aykırı değerlere
        karşı dayanıklıdır.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            Eksik değerlerin doldurulacağı DataFrame.
        fit : bool, varsayılan=True
            True ise imputer'lar yeniden fit edilir (eğitim verisi için).
            False ise mevcut imputer'lar kullanılır (test verisi için).
        
        Döndürür:
        ---------
        pd.DataFrame
            Eksik değerleri doldurulmuş DataFrame.
        """
        df = df.copy()
        
        # Eksik değer istatistiklerini gösteriyoruz
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            print(f"  - Toplam eksik değer sayısı: {total_missing}")
        else:
            print("  - Eksik değer bulunmadı!")
            return df
        
        # Sayısal sütunlar için medyan ile doldurma
        if self.numeric_columns:
            if fit:
                self.numeric_imputer = SimpleImputer(strategy='median')
                df[self.numeric_columns] = self.numeric_imputer.fit_transform(
                    df[self.numeric_columns]
                )
            else:
                df[self.numeric_columns] = self.numeric_imputer.transform(
                    df[self.numeric_columns]
                )
        
        # Kategorik sütunlar için mod ile doldurma
        if self.categorical_columns:
            if fit:
                self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                df[self.categorical_columns] = self.categorical_imputer.fit_transform(
                    df[self.categorical_columns]
                )
            else:
                df[self.categorical_columns] = self.categorical_imputer.transform(
                    df[self.categorical_columns]
                )
        
        print("  ✓ Eksik değerler başarıyla dolduruldu")
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Kategorik değişkenleri sayısal değerlere dönüştürür (Label Encoding).
        
        Her kategorik sütun için ayrı bir LabelEncoder oluşturulur ve
        kategoriler 0'dan başlayan tam sayılara dönüştürülür.
        
        Örnek: ['Eco', 'Business', 'Eco Plus'] -> [1, 0, 2]
        
        Parametreler:
        ------------
        df : pd.DataFrame
            Encode edilecek DataFrame.
        fit : bool, varsayılan=True
            True ise encoder'lar yeniden fit edilir.
        
        Döndürür:
        ---------
        pd.DataFrame
            Kategorik sütunları encode edilmiş DataFrame.
        """
        df = df.copy()
        
        for column in self.categorical_columns:
            if fit:
                # Yeni encoder oluştur ve fit et
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column].astype(str))
                self.label_encoders[column] = encoder
            else:
                # Mevcut encoder'ı kullan
                encoder = self.label_encoders[column]
                # Bilinmeyen kategorileri handle etmek için
                df[column] = df[column].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        
        print(f"  ✓ {len(self.categorical_columns)} kategorik sütun encode edildi")
        return df
    
    def _scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Özellikleri StandardScaler ile normalleştirir (z-score normalization).
        
        Her özellik için ortalama 0, standart sapma 1 olacak şekilde
        dönüşüm uygulanır. Bu, farklı ölçeklerdeki özelliklerin modeli
        eşit şekilde etkilemesini sağlar.
        
        Formül: z = (x - μ) / σ
        
        Parametreler:
        ------------
        X : np.ndarray
            Ölçeklendirilecek özellik matrisi.
        fit : bool, varsayılan=True
            True ise scaler yeniden fit edilir.
        
        Döndürür:
        ---------
        np.ndarray
            Ölçeklendirilmiş özellik matrisi.
        """
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        print("  ✓ Özellikler StandardScaler ile ölçeklendirildi")
        return X_scaled
    
    def _encode_target(self, y: pd.Series) -> np.ndarray:
        """
        Hedef değişkeni (satisfaction) binary formata dönüştürür.
        
        'satisfied' -> 1
        'neutral or dissatisfied' -> 0
        
        Parametreler:
        ------------
        y : pd.Series
            Hedef değişken serisi.
        
        Döndürür:
        ---------
        np.ndarray
            Binary encode edilmiş hedef değişken.
        """
        # Hedef değişkeni binary'e çeviriyoruz
        y_encoded = (y == 'satisfied').astype(int).values
        
        satisfied_count = y_encoded.sum()
        total_count = len(y_encoded)
        dissatisfied_count = total_count - satisfied_count
        
        print(f"  ✓ Hedef değişken encode edildi:")
        print(f"    - Satisfied (1): {satisfied_count:,} ({satisfied_count/total_count*100:.1f}%)")
        print(f"    - Dissatisfied (0): {dissatisfied_count:,} ({dissatisfied_count/total_count*100:.1f}%)")
        
        return y_encoded
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Veri setini işler ve eğitim/test setlerine böler.
        
        Bu metot tüm ön işleme adımlarını sırasıyla uygular:
        1. Gereksiz sütunları atar
        2. Sütun tiplerini belirler
        3. Eksik değerleri doldurur
        4. Kategorik değişkenleri encode eder
        5. Hedef değişkeni ayırır ve encode eder
        6. Veriyi eğitim/test olarak böler
        7. Özellikleri ölçeklendirir
        
        Parametreler:
        ------------
        df : pd.DataFrame
            İşlenecek ham DataFrame.
        
        Döndürür:
        ---------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (X_train, X_test, y_train, y_test) şeklinde dört numpy array.
        
        Kullanım:
        ---------
        >>> preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
        >>> X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
        """
        print("\n" + "=" * 60)
        print("VERİ ÖN İŞLEME BAŞLIYOR")
        print("=" * 60)
        
        df = df.copy()
        
        # Adım 1: Gereksiz sütunları atıyoruz
        print("\n[1/7] Gereksiz sütunlar atılıyor...")
        columns_to_remove = [col for col in self.columns_to_drop if col in df.columns]
        df = df.drop(columns=columns_to_remove, errors='ignore')
        print(f"  ✓ {len(columns_to_remove)} sütun atıldı: {columns_to_remove}")
        
        # Adım 2: Sütun tiplerini belirliyoruz
        print("\n[2/7] Sütun tipleri belirleniyor...")
        self._identify_column_types(df)
        
        # Adım 3: Eksik değerleri dolduruyoruz
        print("\n[3/7] Eksik değerler dolduruluyor...")
        df = self._handle_missing_values(df, fit=True)
        
        # Adım 4: Kategorik değişkenleri encode ediyoruz
        print("\n[4/7] Kategorik değişkenler encode ediliyor...")
        df = self._encode_categorical(df, fit=True)
        
        # Adım 5: Hedef değişkeni ayırıyoruz
        print("\n[5/7] Hedef değişken ayrılıyor ve encode ediliyor...")
        if self.target_column not in df.columns:
            raise ValueError(f"Hedef sütun '{self.target_column}' veri setinde bulunamadı!")
        
        y = self._encode_target(df[self.target_column])
        X = df.drop(columns=[self.target_column]).values
        
        # Adım 6: Veriyi bölüyoruz
        print("\n[6/7] Veri eğitim/test setlerine bölünüyor...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Sınıf dengesini korumak için stratified split
        )
        print(f"  ✓ Eğitim seti: {len(X_train):,} örnek")
        print(f"  ✓ Test seti: {len(X_test):,} örnek")
        
        # Adım 7: Özellikleri ölçeklendiriyoruz
        print("\n[7/7] Özellikler ölçeklendiriliyor...")
        X_train = self._scale_features(X_train, fit=True)
        X_test = self._scale_features(X_test, fit=False)
        
        self._is_fitted = True
        
        print("\n" + "=" * 60)
        print("✓ VERİ ÖN İŞLEME TAMAMLANDI!")
        print("=" * 60)
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Yeni veriyi önceden fit edilmiş parametrelerle dönüştürür.
        
        Bu metot, production ortamında yeni gelen verileri işlemek için
        kullanılır. fit_transform() ile eğitim verisinden öğrenilen
        parametreler (encoder değerleri, scaler ortalaması vb.) kullanılır.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            Dönüştürülecek yeni DataFrame.
        
        Döndürür:
        ---------
        np.ndarray
            İşlenmiş özellik matrisi.
        
        Hatalar:
        --------
        RuntimeError
            Eğer fit_transform() henüz çağrılmamışsa fırlatılır.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Preprocessor henüz fit edilmemiş! "
                "Önce fit_transform() metodunu çağırın."
            )
        
        df = df.copy()
        
        # Gereksiz sütunları at
        df = df.drop(columns=self.columns_to_drop, errors='ignore')
        
        # Hedef sütunu varsa at (prediction için)
        if self.target_column in df.columns:
            df = df.drop(columns=[self.target_column])
        
        # Eksik değerleri doldur
        df = self._handle_missing_values(df, fit=False)
        
        # Kategorik encode
        df = self._encode_categorical(df, fit=False)
        
        # Ölçeklendir
        X = self._scale_features(df.values, fit=False)
        
        return X


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataPreprocessor]:
    """
    Veri ön işleme için kısa yol (shortcut) fonksiyonu.
    
    DataPreprocessor sınıfını kullanarak veriyi hızlıca işler ve
    hem işlenmiş veriyi hem de preprocessor nesnesini döndürür.
    
    Parametreler:
    ------------
    df : pd.DataFrame
        İşlenecek ham DataFrame.
    test_size : float, varsayılan=0.2
        Test seti oranı.
    random_state : int, varsayılan=42
        Rastgele tohum değeri.
    
    Döndürür:
    ---------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataPreprocessor]
        (X_train, X_test, y_train, y_test, preprocessor) tuple'ı.
    
    Kullanım:
    ---------
    >>> X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    """
    preprocessor = DataPreprocessor(test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    return X_train, X_test, y_train, y_test, preprocessor


# Bu modül doğrudan çalıştırıldığında test amaçlı kullanım
if __name__ == "__main__":
    print("Preprocessing Modülü - Test")
    print("-" * 40)
    print("\nBu modülü test etmek için main.py dosyasını çalıştırın.")
    print("Kullanım: python main.py")
