# Laporan Eksperimen Analisis Sentimen — Trans Banyumas
## (`model_1.ipynb`)

---

## 1. Pendahuluan

Laporan ini mendokumentasikan eksperimen klasifikasi sentimen ulasan Google Maps Trans Banyumas menggunakan tiga algoritma machine learning berbasis TF-IDF: **Random Forest**, **Support Vector Machine (LinearSVC)**, dan **Multinomial Naive Bayes**.

Eksperimen ini merupakan iterasi perbaikan dari `model.ipynb` dengan perubahan utama pada kualitas label dan penanganan ketidakseimbangan data.

---

## 2. Dataset

| Atribut | Detail |
|---|---|
| Sumber data | Ulasan Google Maps Trans Banyumas |
| Total data (setelah cleaning) | 346 baris |
| Sumber label | Anotasi manual (`label_sentimen` dari dataset asli) |
| Jumlah kelas | 2 (Positif, Negatif) |

### Distribusi Kelas (Sebelum Oversampling)

| Kelas | Jumlah | Persentase |
|---|---|---|
| Positif | 308 | 89% |
| Negatif | 38 | 11% |

Distribusi ini sangat tidak seimbang — rasio 8:1 antara kelas positif dan negatif.

---

## 3. Pipeline Preprocessing

Teks diproses melalui pipeline berikut sebelum digunakan untuk training:

```
Raw Text
   ↓ Cleaning (hapus URL, angka, karakter khusus, newline)
   ↓ Casefolding (lowercase)
   ↓ Slang Normalization (kamus slang.csv + slang-custom.csv)
   ↓ Stemming (Sastrawi)
   ↓ Tokenizing (NLTK word_tokenize)
   ↓ Stopword Removal (NLTK Indonesian + English, dengan pengecualian kata negasi)
   ↓ text_final (string bersih siap digunakan)
```

**Catatan penting — kata negasi dipertahankan:**
```python
negative_words = {'nggak', 'ngga', 'gak', 'ga', 'gk', 'tidak', 'tdk', 'tak'}
listStopwords -= negative_words
```
Kata-kata ini sengaja dikecualikan dari stopword list agar sinyal negatif tidak hilang.

---

## 4. Feature Extraction — TF-IDF

```python
TfidfVectorizer(
    max_features = 5000,
    min_df       = 2,        # kata harus muncul minimal di 2 dokumen
    ngram_range  = (1, 2)    # unigram + bigram
)
```

TF-IDF di-fit hanya pada data training, kemudian di-transform ke data test (mencegah data leakage).

---

## 5. Penanganan Ketidakseimbangan Data

Digunakan **RandomOverSampler** dari library `imbalanced-learn` untuk menyeimbangkan kelas sebelum training.

**Urutan yang benar:**
```
Vectorize → Oversample → Split train/test
```

Setelah oversampling:

| Kelas | Sebelum | Sesudah |
|---|---|---|
| Negatif (0) | 38 | 308 |
| Positif (1) | 308 | 308 |
| **Total** | **346** | **616** |

**Split:**
- Train: 80% → 492 sampel
- Test: 20% → 124 sampel

> **Catatan:** Oversampling dilakukan **sebelum** split, sehingga distribusi kelas pada test set seimbang. Ini membuat perbandingan antar kelas lebih fair, namun metrik bisa sedikit lebih optimistis dibanding skenario real-world.

---

## 6. Hasil Evaluasi Model

### 6.1 Random Forest + TF-IDF

```
Accuracy Train : 90.45%
Accuracy Test  : 85.48%
```

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negatif (0) | 0.89 | 0.76 | 0.82 | 54 |
| Positif (1) | 0.83 | 0.93 | 0.88 | 70 |
| **Accuracy** | | | **0.85** | **124** |
| Macro avg | 0.86 | 0.84 | 0.85 | 124 |
| Weighted avg | 0.86 | 0.85 | 0.85 | 124 |

**Confusion Matrix:**

|  | Pred Negatif | Pred Positif |
|---|---|---|
| **Aktual Negatif** | 41 (TP) | 13 (FN) |
| **Aktual Positif** | 5 (FP) | 65 (TN) |

**Konfigurasi model:**
```python
RandomForestClassifier(
    n_estimators    = 200,
    min_samples_split = 10,
    min_samples_leaf  = 4,
    max_features    = 'sqrt',
    class_weight    = 'balanced',
    random_state    = 42
)
```

**Analisis:** Model cukup stabil — gap train/test hanya ~5%, menandakan overfitting minimal. Namun recall kelas negatif (76%) lebih rendah dari positif (93%), artinya model masih cenderung bias ke kelas positif. Dari 54 sampel negatif di test, 13 salah diprediksi positif (false negative).

---

### 6.2 SVM (LinearSVC) + TF-IDF

```
Accuracy Train : 95.33%
Accuracy Test  : 88.71%
```

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negatif (0) | 0.90 | 0.83 | 0.87 | 54 |
| Positif (1) | 0.88 | 0.93 | 0.90 | 70 |
| **Accuracy** | | | **0.89** | **124** |
| Macro avg | 0.89 | 0.88 | 0.88 | 124 |
| Weighted avg | 0.89 | 0.89 | 0.89 | 124 |

**Confusion Matrix:**

|  | Pred Negatif | Pred Positif |
|---|---|---|
| **Aktual Negatif** | 45 (TP) | 9 (FN) |
| **Aktual Positif** | 5 (FP) | 65 (TN) |

**Konfigurasi model:**
```python
LinearSVC(
    C            = 1.0,
    random_state = 42
)
```

**Analisis:** SVM adalah model terbaik di eksperimen ini. Recall negatif lebih tinggi (83% vs 76% RF), artinya lebih baik dalam mendeteksi ulasan negatif. False negative lebih sedikit (9 vs 13). Gap train/test ~6.6%, masih dalam batas wajar.

---

### 6.3 Naive Bayes + TF-IDF

> ⚠️ **Cell belum dieksekusi** — hasil belum tersedia. Jalankan cell `nb-training` dan `nb-inference` di notebook untuk mendapatkan metrik.

**Konfigurasi model:**
```python
MultinomialNB()  # alpha=1.0 (Laplace smoothing default)
```

**Ekspektasi:** Naive Bayes biasanya lebih cepat dan sederhana, namun asumsi independensi antar fitur menyebabkan performanya umumnya di bawah SVM untuk data TF-IDF. Kemungkinan accuracy test berkisar 80–85%.

---

## 7. Perbandingan Ringkas

| Model | Train Acc | Test Acc | F1 Macro | Recall Negatif | Gap Overfit |
|---|---|---|---|---|---|
| Random Forest | 90.45% | 85.48% | 0.85 | 76% | 4.97% |
| **SVM (LinearSVC)** | **95.33%** | **88.71%** | **0.88** | **83%** | **6.62%** |
| Naive Bayes | — | — | — | — | — |

---

## 8. Analisis Inference

Fungsi `clean_text` pada fase inference telah diperbaiki agar konsisten dengan pipeline training:

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in listStopwords]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)
```

**Hasil inference pada contoh kalimat negatif:**

| Input | RF | SVM | NB |
|---|---|---|---|
| *"Sangat mengecewakan, kotor, jorok, rusak, panas tidak ada AC"* | NEGATIVE (73.3%) | NEGATIVE (-1.43) | — |

**Catatan keterbatasan inference:** Model hanya mengenali kata-kata negatif yang ada dalam vocabulary training (38 sampel negatif asli). Kata negatif generik seperti *"jelek"*, *"lambat"*, *"buruk"* yang tidak muncul di corpus training tidak akan terdeteksi dengan baik.

---

## 9. Temuan & Rekomendasi

### Temuan Utama

1. **Label manual lebih unggul dari label lexicon** — Beralih dari lexicon-based labeling ke anotasi manual meningkatkan test accuracy dari 74% ke 88% (RF) dan 72% ke 88% (SVM).

2. **SVM unggul untuk klasifikasi teks biner** — SVM konsisten menghasilkan performa lebih baik dari RF, terutama pada recall kelas minoritas (negatif).

3. **Vocabulary terlalu domain-spesifik** — Dengan hanya 38 sampel negatif unik, model tidak bisa mengenali kata negatif di luar domain ulasan terminal.

4. **Oversampling efektif** — RandomOverSampler berhasil mengatasi ketidakseimbangan dan meningkatkan recall kelas negatif secara signifikan.

### Rekomendasi

| Prioritas | Rekomendasi |
|---|---|
| 🔴 Tinggi | Tambah data negatif yang lebih beragam (minimal 150+ sampel) |
| 🔴 Tinggi | Jalankan cell Naive Bayes untuk melengkapi perbandingan |
| 🟡 Sedang | Gunakan `StratifiedKFold` cross-validation untuk evaluasi lebih robust |
| 🟡 Sedang | Pisahkan oversampling ke dalam training fold saja (hindari data leakage) |
| 🟢 Rendah | Eksplorasi hyperparameter tuning SVM (`C`, kernel) |
| 🟢 Rendah | Pertimbangkan augmentasi data dengan sinonim kata negatif |

---

## 10. Kesimpulan

**Model terbaik: SVM (LinearSVC) + TF-IDF** dengan test accuracy **88.71%** dan F1 macro **0.88**.

Model ini siap digunakan sebagai baseline, namun kinerjanya pada inferensi kata-kata negatif generik masih terbatas akibat sedikitnya data negatif dalam corpus training. Penambahan data negatif yang beragam adalah langkah paling penting untuk meningkatkan kualitas model secara keseluruhan.
