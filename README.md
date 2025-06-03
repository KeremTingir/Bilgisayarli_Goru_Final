# OpenCV ile Görüntü İşleme Projeleri

Bu repo, OpenCV ve Python kullanılarak geliştirilmiş çeşitli görüntü işleme uygulamalarını içermektedir. Projeler arasında belge tarama, optik form okuma (OMR) ve çeşitli nesnelerin (mercimek, nohut, pirinç) sayımı gibi farklı görevler bulunmaktadır.

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Dosya Açıklamaları](#dosya-açıklamaları)
  - [scanner.py](#scannerpy)
  - [optik_okuma.py](#optik_okumapy)
  - [mercimek_sayisi.py](#mercimek_sayisipy)
  - [nohut_sayisi.py](#nohut_sayisipy)
  - [pirinc_sayisi.py](#pirinc_sayisipy)
  - [goruntu_isleme_adimlari.py](#goruntu_isleme_adımlaripy)
  - [optik_okuma_sonuc_grafik.py](#optik_okuma_sonuc_grafikpy)
- [Dizin Yapısı](#dizin-yapısı)
- [Olası Geliştirmeler](#olası-geliştirmeler)
- [Katkıda Bulunma](#katkıda-bulunma)

## Genel Bakış

Bu proje koleksiyonu, temel ve orta seviye görüntü işleme tekniklerini pratik örneklerle göstermeyi amaçlamaktadır. Her bir betik, belirli bir sorunu çözmek için farklı OpenCV fonksiyonlarını ve algoritmalarını kullanır.

## Özellikler

*   **Belge Tarama:** Bir belgenin görüntüsünden kenarlarını tespit etme, perspektifini düzeltme ve okunabilirliğini artırma.
*   **Optik Form Okuma (OMR):** İşaretlenmiş cevap şıklarını algılayarak bir optik formu değerlendirme.
*   **Nesne Sayımı:** Farklı türdeki küçük nesnelerin (mercimek, nohut, pirinç) bir görüntüdeki sayısını belirleme.
*   **Görüntü Ön İşleme:** Gri tonlama, bulanıklaştırma, eşikleme, morfolojik operasyonlar gibi adımlar.
*   **Kontur Analizi:** Nesneleri tespit etmek ve özelliklerini çıkarmak için konturları kullanma.
*   **Görselleştirme:** Matplotlib kütüphanesi ile işleme adımlarını ve sonuçlarını görselleştirme.

## Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Depoyu Klonlayın:**
    ```bash
    git clone https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git
    cd REPO_ADINIZ
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    Python 3.x sürümünün kurulu olduğundan emin olun.
    ```bash
    pip install opencv-python numpy matplotlib
    ```

3.  **`images` ve `results` Klasörleri:**
    Betiklerin çoğu `images/` klasöründen girdi görüntülerini okur. `scanner.py` betiği çıktılarını `results/Bolum1/` klasörüne kaydeder. Bu klasörlerin mevcut olduğundan emin olun veya gerekirse oluşturun:
    ```bash
    mkdir images
    mkdir -p results/Bolum1
    ```
    Örnek görüntüleri `images/` klasörüne yerleştirin.

## Kullanım

Her bir Python betiğini komut satırından çalıştırabilirsiniz. Betiklerin çoğu, işledikleri görüntülerin yolunu kod içinde belirtir (`images/dosya_adi.png` gibi).

Örnek çalıştırma komutları:

```bash
python scanner.py
python optik_okuma.py
python mercimek_sayisi.py
python nohut_sayisi.py
python pirinc_sayisi.py
python goruntu_isleme_adimlari.py
python optik_okuma_sonuc_grafik.py