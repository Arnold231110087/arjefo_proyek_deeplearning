# PadiCek — Deteksi Penyakit Daun Padi Berbasis CNN

PadiCek adalah website yang mengenali kondisi daun padi dari sebuah foto. Model yang dipakai adalah **Convolutional Neural Network (CNN)** klasifikasi multikelas (output softmax, loss categorical cross-entropy, metrik accuracy) dengan Data Augmentation dan Dropout, dan dilatih dari awal. Website juga menampilkan **Grad-CAM** (bagian daun yang menjadi dasar keputusan model) dan **ambang keyakinan** (foto yang tidak meyakinkan ditandai, tidak langsung diberi diagnosis).

Proyek UTS mata kuliah **Pembelajaran Mendalam**, Kelompok **Arjefo**, Universitas Mikroskil, Semester Ganjil TA 2026-2027.

- **Website:** https://padicek.streamlit.app
- **Dataset:** [Rice Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)
- **Video demo:** _(menyusul)_

## Kelas yang dikenali

| Kelas (dataset) | Nama Indonesia |
|---|---|
| Bacterial Leaf Blight | Hawar Daun Bakteri |
| Brown Spot | Bercak Coklat |
| Leaf Blast | Blas Daun |
| Leaf scald | Hangus Daun |
| Sheath Blight | Hawar Pelepah |
| Healthy Rice Leaf | Daun Sehat |

## Hasil model (data uji)

Model yang dipakai website adalah skenario **S1**. Model ini dipilih otomatis dari tiga skenario berdasarkan akurasi validasi tertinggi, dan data uji tidak dipakai untuk memilih.

| Metrik | Nilai |
|---|---|
| Akurasi | 92,24% |
| Precision (macro) | 92,22% |
| Recall (macro) | 92,22% |
| F1-score (macro) | 92,20% |
| Loss | 0,2482 |
| Jumlah parameter | 423.526 |

Data: 3.798 gambar setelah pembersihan, dibagi 2.719 latih / 538 validasi / 541 uji. Pembagian dilakukan secara stratified berbasis grup supaya gambar yang mirip tidak bocor ke data uji. Rincian lengkapnya ada di tab **Performa Model** pada website dan di laporan.

## Struktur repository

```
arjefo_proyek_deeplearning/
├── Arjefo_Proyek_DeepLearning.ipynb   # Notebook Google Colab: data, EDA, preprocessing, training, evaluasi, fitur unik
├── README.md
└── padicek/                           # Folder website (main file Streamlit: padicek/app.py)
    ├── app.py                         # Website Streamlit: deteksi, Grad-CAM, ambang keyakinan, performa model
    ├── fitur_unik.py                  # Fokus daun, Grad-CAM, dan ambang keyakinan (dipakai notebook DAN website)
    ├── info_penyakit.py               # Nama, penyebab, gejala, dan penanganan setiap penyakit
    ├── requirements.txt               # Library untuk website
    ├── .streamlit/config.toml         # Tema website
    ├── model/
    │   ├── padicek_model.keras        # Model final hasil training
    │   ├── labels.json                # Nama kelas, ukuran input, dan nilai ambang keyakinan
    │   └── skenario/                  # Model ketiga skenario (S1, S2, S3)
    └── hasil/
        ├── metrics.json               # Semua angka evaluasi (dipakai website dan laporan)
        └── *.png                      # Grafik EDA dan evaluasi
```

## Alur notebook

Notebook `Arjefo_Proyek_DeepLearning.ipynb` dijalankan di Google Colab (Runtime → T4 GPU → *Run all*):

| Langkah | Isi |
|---|---|
| 1–3 | Menyiapkan file pendukung, konfigurasi, mengambil dataset dari Kaggle |
| 4 | Preprocessing a: pembersihan data (duplikat, file rusak, konflik label, grup gambar mirip) |
| 5 | Exploratory Data Analysis (EDA) |
| 6–9 | Preprocessing b–e: resize dan fokus daun, augmentasi, outlier dan class weight, pembagian data tanpa kebocoran |
| 10–11 | Arsitektur CNN, training tiga skenario, dan pemilihan model final |
| 12–14, 17–18 | Hasil evaluasi a–e: metrik, nilai pada data uji, grafik, perbandingan skenario, analisis kesalahan |
| 15–16 | Fitur unik: Grad-CAM dan ambang keyakinan |
| 19–21 | Menyimpan model, mengunduh hasil, dan menjalankan website dari Colab |

## Menjalankan website di komputer sendiri

Butuh Python 3.10–3.14.

```bash
git clone https://github.com/arnold231110087/arjefo_proyek_deeplearning.git
cd arjefo_proyek_deeplearning/padicek
pip install -r requirements.txt
python -m streamlit run app.py
```

Website terbuka di `http://localhost:8501`.

## Deploy ke Streamlit Community Cloud

1. Login ke [share.streamlit.io](https://share.streamlit.io) dengan akun GitHub.
2. **Create app**, lalu isi: repository `arnold231110087/arjefo_proyek_deeplearning`, branch `main`, main file path `padicek/app.py`.
3. Klik **Deploy**. Server memasang library dari `padicek/requirements.txt` (sekitar 5–10 menit untuk pertama kali).

Catatan:
- `requirements.txt` otomatis menyesuaikan versi Python server: tensorflow-cpu 2.21.0 untuk Python 3.10–3.13, dan tensorflow-cpu 2.22.0rc0 untuk Python 3.14. Model menghasilkan prediksi yang sama di kedua versi.
- `app.py` memindahkan folder kerja ke folder `padicek/` (`os.chdir`), sehingga folder `model/` dan `hasil/` tetap ditemukan walaupun Streamlit Cloud menjalankan aplikasi dari folder utama repository.

## Cara memakai website

1. Buka tab **Deteksi**, lalu unggah foto daun padi (JPG, PNG, WEBP, JFIF, BMP, GIF, atau TIFF) atau ambil foto dengan kamera.
2. Jika foto berisi banyak latar (sawah, tangan, teks), potong tepi foto agar daun yang sakit memenuhi bingkai.
3. Lihat hasilnya: nama penyakit, tingkat keyakinan, probabilitas setiap kelas, dan saran penanganan.
4. Pilih tampilan **Grad-CAM**, **Fokus daun**, atau **Foto asli** untuk melihat bagian daun yang dianalisis model.

Hasil paling akurat jika satu helai daun yang sakit terlihat jelas, karena model hanya mengenal gaya foto yang ada di dataset.

## Anggota kelompok Arjefo

| NIM | Nama | Peran |
|---|---|---|
| 231110087 | Arnold Jefverson | Ketua, Model Engineer & Web Developer |
| 231110232 | Valerio Liuz Kienata | Data Engineer |
| 231112175 | Richardo Lieberio | Preprocessing & Evaluasi |
| 231111520 | Rendy | Fitur Unik |
| 231110804 | Alvin Ong | Dokumentasi |
