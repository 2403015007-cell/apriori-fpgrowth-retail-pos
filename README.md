# apriori-fpgrowth-retail-pos
Perbandingan Apriori dan FP-Growth, uji skalabilitas, dan uji permutasi pada dataset Retail POS (Kaggle)
# Apriori vs FP-Growth pada Retail POS

Kode eksperimen untuk naskah "Perbandingan Apriori dan FP-Growth dengan
Pemeriksaan Data Acak pada Dataset Sintetis Transaksi Produk Pangan".

## Data
Unduh retail_pos_basket_data.csv dari Kaggle (lisensi MIT):
https://www.kaggle.com/datasets/arunsworkspace/retail-pos-dataset-for-market-basket-analysis
Letakkan di folder yang sama dengan skrip.

## Cara menjalankan
pip install mlxtend==0.25.0 pandas scipy numpy
python eksperimen_revisi.py --csv retail_pos_basket_data.csv --seed 42

Keluaran: hasil_ulangan_waktu.csv, hasil_skalabilitas.csv, hasil_permutasi.csv

## Lingkungan
Python 3.13, pandas 3.0.5, mlxtend 0.25.0, Linux (2 vCPU, ~8 GB RAM).
Waktu dan memori bergantung pada mesin. Uji permutasi (1.000 data acak)
memakan waktu cukup lama.

## Lisensi
Kode: MIT. Data mentah mengikuti lisensi dataset di Kaggle.
