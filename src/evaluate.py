# ============================================================================
# DEĞERLENDİRME MODÜLÜ - evaluate.py
# ============================================================================
# Bu modül, eğitilmiş modellerin performansını ölçmek için gerekli
# metrik hesaplama ve görselleştirme fonksiyonlarını içerir.
# ============================================================================

"""
Model Değerlendirme Modülü

Bu modül, makine öğrenmesi modellerinin performansını değerlendirmek
için çeşitli metrikler ve görselleştirmeler sağlar.

Hesaplanan Metrikler:
    - Accuracy (Doğruluk)
    - Precision (Kesinlik)
    - Recall (Duyarlılık)
    - F1-Score
    - Confusion Matrix (Karışıklık Matrisi)

Fonksiyonlar:
    - evaluate_model: Tek bir modeli değerlendirir
    - print_classification_report: Detaylı sınıflandırma raporu yazdırır
    - plot_confusion_matrix: Karışıklık matrisini görselleştirir
    - compare_models: Birden fazla modelin sonuçlarını karşılaştırır
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import seaborn as sns
from .model import SatisfactionModel


def evaluate_model(
    model: SatisfactionModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Eğitilmiş modelin test verisi üzerindeki performansını değerlendirir.
    
    Bu fonksiyon, modelin tahminlerini alır ve çeşitli sınıflandırma
    metriklerini hesaplar. Sonuçlar hem sayısal değerler hem de
    görsel formatta sunulabilir.
    
    Parametreler:
    ------------
    model : SatisfactionModel
        Değerlendirilecek eğitilmiş model.
    
    X_test : np.ndarray
        Test özellikleri.
    
    y_test : np.ndarray
        Gerçek test etiketleri.
    
    verbose : bool, varsayılan=True
        True ise sonuçlar konsola yazdırılır.
    
    Döndürür:
    ---------
    Dict[str, Any]
        Tüm metrikleri içeren sözlük:
        - 'accuracy': Doğruluk skoru
        - 'precision': Kesinlik skoru
        - 'recall': Duyarlılık skoru
        - 'f1_score': F1 skoru
        - 'confusion_matrix': Karışıklık matrisi
        - 'predictions': Model tahminleri
        - 'roc_auc': ROC-AUC skoru (varsa)
    
    Örnek:
    ------
    >>> metrics = evaluate_model(model, X_test, y_test)
    >>> print(f"F1 Skoru: {metrics['f1_score']:.4f}")
    """
    model_name = model.get_model_name()
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MODEL DEĞERLENDİRMESİ: {model_name}")
        print(f"{'=' * 60}")
    
    # Tahminleri alıyoruz
    y_pred = model.predict(X_test)
    
    # Temel metrikleri hesaplıyoruz
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # ROC-AUC hesaplamayı deniyoruz (predict_proba varsa)
    roc_auc = None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        pass  # predict_proba desteklenmiyorsa atla
    
    # Sonuç sözlüğü
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred,
        'roc_auc': roc_auc
    }
    
    if verbose:
        print_metrics(results)
    
    return results


def print_metrics(metrics: Dict[str, Any]) -> None:
    """
    Hesaplanan metrikleri formatlanmış şekilde konsola yazdırır.
    
    Bu fonksiyon, evaluate_model() tarafından döndürülen metrik
    sözlüğünü alır ve okunabilir bir formatta ekrana yazdırır.
    
    Parametreler:
    ------------
    metrics : Dict[str, Any]
        Metrikleri içeren sözlük.
    """
    print("\n📊 PERFORMANS METRİKLERİ:")
    print("-" * 40)
    
    # Ana metrikler
    print(f"  • Doğruluk (Accuracy):     {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  • Kesinlik (Precision):    {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)")
    print(f"  • Duyarlılık (Recall):     {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)")
    print(f"  • F1-Skoru:                {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.2f}%)")
    
    if metrics.get('roc_auc'):
        print(f"  • ROC-AUC:                 {metrics['roc_auc']:.4f}  ({metrics['roc_auc']*100:.2f}%)")
    
    # Karışıklık matrisi
    cm = metrics['confusion_matrix']
    print("\n📋 KARIŞIKLIK MATRİSİ:")
    print("-" * 40)
    print(f"                    Tahmin")
    print(f"                  Neg    Pos")
    print(f"  Gerçek  Neg  [{cm[0][0]:5d}  {cm[0][1]:5d}]")
    print(f"          Pos  [{cm[1][0]:5d}  {cm[1][1]:5d}]")
    
    # Karışıklık matrisinin yorumu
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Açıklama:")
    print(f"    - True Negative (Doğru Negatif):  {tn:,}")
    print(f"    - False Positive (Yanlış Pozitif): {fp:,}")
    print(f"    - False Negative (Yanlış Negatif): {fn:,}")
    print(f"    - True Positive (Doğru Pozitif):  {tp:,}")


def print_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> str:
    """
    Detaylı sınıflandırma raporunu yazdırır ve döndürür.
    
    Scikit-learn'ün classification_report fonksiyonunu kullanarak
    her sınıf için ayrı ayrı precision, recall ve f1-score değerlerini
    gösterir.
    
    Parametreler:
    ------------
    y_test : np.ndarray
        Gerçek etiketler.
    
    y_pred : np.ndarray
        Model tahminleri.
    
    target_names : List[str], opsiyonel
        Sınıf isimleri. Varsayılan: ['Dissatisfied', 'Satisfied']
    
    Döndürür:
    ---------
    str
        Formatlanmış sınıflandırma raporu.
    """
    if target_names is None:
        target_names = ['Dissatisfied', 'Satisfied']
    
    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        digits=4
    )
    
    print("\n📑 SINIFLANDIRMA RAPORU:")
    print("-" * 60)
    print(report)
    
    return report


def plot_confusion_matrix(
    metrics: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Karışıklık matrisini ısı haritası olarak görselleştirir.
    
    Seaborn kütüphanesi kullanılarak renkli ve etiketli bir
    karışıklık matrisi grafiği oluşturulur.
    
    Parametreler:
    ------------
    metrics : Dict[str, Any]
        evaluate_model() tarafından döndürülen metrik sözlüğü.
    
    save_path : str, opsiyonel
        Grafiğin kaydedileceği dosya yolu. None ise kaydedilmez.
    
    figsize : Tuple[int, int], varsayılan=(8, 6)
        Grafik boyutu (genişlik, yükseklik) inç cinsinden.
    
    Örnek:
    ------
    >>> metrics = evaluate_model(model, X_test, y_test)
    >>> plot_confusion_matrix(metrics, save_path='confusion_matrix.png')
    """
    cm = metrics['confusion_matrix']
    model_name = metrics.get('model_name', 'Model')
    
    # Grafik oluşturuyoruz
    plt.figure(figsize=figsize)
    
    # Isı haritası çiziyoruz
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Dissatisfied', 'Satisfied'],
        yticklabels=['Dissatisfied', 'Satisfied'],
        annot_kws={'size': 14}
    )
    
    plt.title(f'Karışıklık Matrisi - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Karışıklık matrisi kaydedildi: {save_path}")
    
    plt.show()


def plot_roc_curve(
    model: SatisfactionModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    ROC eğrisini çizer.
    
    Receiver Operating Characteristic (ROC) eğrisi, modelin
    farklı eşik değerlerinde gösterdiği performansı görselleştirir.
    
    Parametreler:
    ------------
    model : SatisfactionModel
        Değerlendirilecek model.
    
    X_test : np.ndarray
        Test özellikleri.
    
    y_test : np.ndarray
        Gerçek test etiketleri.
    
    save_path : str, opsiyonel
        Grafiğin kaydedileceği dosya yolu.
    
    figsize : Tuple[int, int], varsayılan=(8, 6)
        Grafik boyutu.
    """
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        print("  ! Bu model ROC eğrisi için olasılık tahmini desteklemiyor.")
        return
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=figsize)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Eğrisi (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Rastgele Sınıflandırıcı')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)', fontsize=12)
    plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)', fontsize=12)
    plt.title(f'ROC Eğrisi - {model.get_model_name()}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ ROC eğrisi kaydedildi: {save_path}")
    
    plt.show()


def compare_models(
    results_list: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Birden fazla modelin sonuçlarını karşılaştırır ve görselleştirir.
    
    Bu fonksiyon, farklı modellerin performans metriklerini
    yan yana bar grafiği olarak gösterir.
    
    Parametreler:
    ------------
    results_list : List[Dict[str, Any]]
        Her model için evaluate_model() sonuçlarının listesi.
    
    save_path : str, opsiyonel
        Grafiğin kaydedileceği dosya yolu.
    
    Örnek:
    ------
    >>> results = [evaluate_model(m, X_test, y_test) for m in models]
    >>> compare_models(results)
    """
    if not results_list:
        print("Karşılaştırılacak sonuç bulunamadı.")
        return
    
    # Model isimlerini ve metrikleri çıkarıyoruz
    model_names = [r['model_name'] for r in results_list]
    accuracies = [r['accuracy'] for r in results_list]
    precisions = [r['precision'] for r in results_list]
    recalls = [r['recall'] for r in results_list]
    f1_scores = [r['f1_score'] for r in results_list]
    
    # Bar grafiği için veri hazırlığı
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Doğruluk', color='#2ecc71')
    bars2 = ax.bar(x - 0.5*width, precisions, width, label='Kesinlik', color='#3498db')
    bars3 = ax.bar(x + 0.5*width, recalls, width, label='Duyarlılık', color='#e74c3c')
    bars4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Skoru', color='#9b59b6')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title('Model Karşılaştırması', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bar değerlerini üzerine yazıyoruz
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Karşılaştırma grafiği kaydedildi: {save_path}")
    
    plt.show()
    
    # Özet tablo
    print("\n📊 MODEL KARŞILAŞTIRMA TABLOSU:")
    print("-" * 70)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    
    for r in results_list:
        print(f"{r['model_name']:<25} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
              f"{r['recall']:<12.4f} {r['f1_score']:<12.4f}")
    
    print("-" * 70)


# Bu modül doğrudan çalıştırıldığında bilgi verir
if __name__ == "__main__":
    print("Evaluate Modülü - Bilgi")
    print("-" * 40)
    print("\nBu modül model değerlendirmesi için kullanılır.")
    print("Kullanım için main.py dosyasını çalıştırın.")
