# Implementasi Algoritma K-Nearest Neighbor dan Naive Bayes Sebagai Prediktor Kelas Harga Ponsel

Repository ini dibuat untuk memenuhi Tugas Besar 2 Intelegensia Buatan IF3170 untuk melakukan prediksi kelas harga ponsel

## Hasil prediksi
Accuracy score prediksi yang didapatkan berdasarkan metode K-Nearest Neighbor, Naive Bayes dan Gaussian Naive Bayes adalah:
|Classifier|Accuracy score|
|----|-------|
|KNN SKLearn |0.801667 |
|KNN | 0.801667 |
|Naive-Bayes SKLearn (GaussianNB) | 0.778333 |
|Naive-Bayes Categorical | 0.728333 |
|Naive-Bayes Discretization | 0.72 |
|Naive Bayes Gaussian | 0.778333 |
|Naive Bayes Kernel Density Estimation| 0.776667 |

Accuracy di atas dilakukan tanpa melakukan preprocessing (kecuali menghilangkan data outlier) dengan train set : 1400 baris dan valid set: 600 baris

## Requirement
Library yang dibutuhkan pada program ini adalah:
* Pandas
* Numpy
* Scikit-Learn
* Scipy

## Anggota
|Nama|Kontak|Github|
|----|-------|------|
|Manuella Ivana Uli Sianipar | 13521051@std.stei.itb.ac.id| <a href="https://www.github.com/manuellaiv">@manuellaiv</a>|
|Yobel Dean Christopher | 13521067@std.stei.itb.ac.id |<a href="https://www.github.com/yobeldc">@yobeldc</a>|
|Jeremya Dharmawan Raharjo | 13521131@std.stei.itb.ac.id|<a href="https://www.github.com/jejejery">@jejejery</a>|
|Cetta Reswara Parahita | 13521133@std.stei.itb.ac.id|<a href="https://www.github.com/CettaReswara">@CettaReswara</a>|
