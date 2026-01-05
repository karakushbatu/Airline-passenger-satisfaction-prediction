# ============================================================================
# SRC PAKETİ - Airline Passenger Satisfaction Prediction
# ============================================================================
# Bu modül, veri işleme, model eğitimi ve değerlendirme için gerekli tüm
# alt modülleri içerir. Modüler bir yapı kullanarak kodun okunabilirliğini
# ve bakımını kolaylaştırıyoruz.
# ============================================================================

"""
Airline Passenger Satisfaction Prediction Projesi - Kaynak Modülleri

Bu paket aşağıdaki modülleri içerir:
    - data_loader: Veri setini yükleme işlemleri
    - preprocessing: Veri ön işleme (encoding, scaling, splitting)
    - model: Makine öğrenmesi model sınıfları
    - train: Model eğitim fonksiyonları
    - evaluate: Model değerlendirme metrikleri

Kullanım Örneği:
    from src.data_loader import load_data
    from src.preprocessing import preprocess_data
    from src.model import SatisfactionModel
"""

# Alt modülleri dışa aktarıyoruz
from .data_loader import load_data
from .preprocessing import DataPreprocessor
from .model import SatisfactionModel
from .train import train_models
from .evaluate import evaluate_model

# Paketin versiyonu
__version__ = "1.0.0"

# Dışa aktarılan tüm bileşenler
__all__ = [
    "load_data",
    "DataPreprocessor", 
    "SatisfactionModel",
    "train_models",
    "evaluate_model"
]
