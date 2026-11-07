import json
import os

import numpy as np
import streamlit as st
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import fitur_unik as fu
from info_penyakit import cari_info
 
MODEL_PATH = "model/padicek_model.keras"
LABEL_PATH = "model/labels.json"
METRICS_PATH = "hasil/metrics.json"
 
st.set_page_config(page_title="PadiCek – Deteksi Penyakit Daun Padi", page_icon="🌾", layout="centered")
 
st.markdown(
    """
    <style>
      .hasil-box {border-radius: 12px; padding: 1.1rem 1.3rem; margin: .6rem 0 1rem;}
      .hasil-sakit {background: #fdf1e6; border: 1px solid #e8b27a;}
      .hasil-sehat {background: #ebf6ec; border: 1px solid #8cc895;}
      .hasil-ragu {background: #f2f2f2; border: 1px dashed #9a9a9a;}
      .hasil-label {font-size: .85rem; color: #5b5b5b; margin: 0;}
      .hasil-nama {font-size: 1.6rem; font-weight: 700; margin: .1rem 0 .2rem; color: #1e1e1e;}
      .hasil-latin {font-style: italic; color: #5b5b5b; margin: 0;}
      @media (prefers-color-scheme: dark) {
        .hasil-sakit {background: #3a2a1a; border-color: #8a5a2b;}
        .hasil-sehat {background: #1c3320; border-color: #3f7a48;}
        .hasil-ragu {background: #2a2a2a; border-color: #777;}
        .hasil-nama {color: #f2f2f2;} .hasil-label, .hasil-latin {color: #c9c9c9;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
@st.cache_resource(show_spinner="Memuat model CNN...")
def muat_model():
    import tensorflow as tf  # diimpor di sini agar halaman tampil lebih cepat
 
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABEL_PATH) as f:
        meta = json.load(f)
    return (model, fu.buat_model_gradcam(model), meta["class_names"], int(meta.get("img_size", 224)),
            float(meta.get("ambang_keyakinan", fu.AMBANG_DEFAULT)), float(meta.get("ambang_minimum", fu.AMBANG_MINIMUM)))
 
 
@st.cache_data
def muat_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None
 
def siapkan(gambar: Image.Image, img_size: int, potong=(0, 0, 0, 0)):
    """potong = persen dipotong dari (kiri, atas, kanan, bawah) agar daun memenuhi bingkai.
    Mengembalikan (input model img_size x img_size, foto asli setelah dipotong untuk ditampilkan)."""
    w, h = gambar.size
    kiri, atas, kanan, bawah = potong
    kotak = (int(w * kiri / 100), int(h * atas / 100), int(w * (1 - kanan / 100)), int(h * (1 - bawah / 100)))
    if kotak[2] - kotak[0] > 8 and kotak[3] - kotak[1] > 8:
        gambar = gambar.crop(kotak)
    # Input CNN wajib berukuran tetap (sama seperti saat training); foto asli TIDAK diubah untuk tampilan.
    return fu.siapkan_gambar(gambar, img_size), gambar

st.title("🌾 PadiCek")
st.caption("Deteksi penyakit daun padi dari foto menggunakan Convolutional Neural Network (CNN)")
 
if not os.path.exists(MODEL_PATH):
    st.error(
        "File model belum ada. Jalankan training terlebih dahulu (jalankan_padicek.py / train.py / notebook), "
        "lalu pastikan folder `model/` dan `hasil/` berada di folder yang sama dengan app.py."
    )
    st.stop()
 
model, model_cam, CLASS_NAMES, IMG_SIZE, AMBANG, MINIMUM = muat_model()
metrics = muat_metrics()
 
tab_deteksi, tab_performa, tab_info, tab_tentang = st.tabs(
    ["🔍 Deteksi", "📊 Performa Model", "📚 Info Penyakit", "ℹ️ Tentang"]
)
 
with tab_deteksi:
    st.write("Unggah foto daun padi (dari HP, kamera, atau internet), atau ambil foto langsung dengan kamera. "
             "Hasil paling akurat jika **satu helai daun** yang sakit terlihat jelas dan memenuhi sebagian besar foto.")
    sumber = st.radio("Sumber foto", ["Unggah file", "Kamera"], horizontal=True, label_visibility="collapsed")
    berkas = (st.file_uploader("Pilih foto daun padi", type=fu.FORMAT_GAMBAR,
                               help="Format: " + ", ".join(f.upper() for f in fu.FORMAT_GAMBAR))
              if sumber == "Unggah file" else st.camera_input("Ambil foto daun padi"))
 
    if berkas is not None:
        try:
            gambar = fu.buka_gambar(berkas)                  
        except Exception:
            st.error("File tidak dapat dibaca sebagai gambar. Coba simpan ulang gambar sebagai JPG lalu unggah lagi.")
            st.stop()
 
        with st.expander("✂️ Potong foto agar daun memenuhi bingkai (opsional)"):
            st.caption("Foto dari internet sering memuat banyak latar (sawah, tanah, langit). Potong tepi foto "
                       "sampai daun yang sakit berada di tengah dan terlihat besar.")
            c1, c2 = st.columns(2)
            p_kiri = c1.slider("Potong kiri (%)", 0, 45, 0)
            p_kanan = c2.slider("Potong kanan (%)", 0, 45, 0)
            p_atas = c1.slider("Potong atas (%)", 0, 45, 0)
            p_bawah = c2.slider("Potong bawah (%)", 0, 45, 0)
        arr, foto = siapkan(gambar, IMG_SIZE, (p_kiri, p_atas, p_kanan, p_bawah))
        tampil = foto.copy()
        tampil.thumbnail((1600, 1600))
        tampil = np.asarray(tampil)
 
        with st.spinner("Menganalisis daun..."):
            cam, prob = fu.gradcam(model_cam, arr)                 # sekali jalan: prediksi + Grad-CAM
        status, yakin = fu.status_prediksi(prob, AMBANG, MINIMUM)  # fitur ambang keyakinan (3 tingkat)

        urut = np.argsort(prob)[::-1]
        info = cari_info(CLASS_NAMES[urut[0]])
 
        kiri, kanan = st.columns([1.2, 1], gap="large")
        with kiri:
            mode = st.radio("Tampilan", ["Grad-CAM", "Fokus daun", "Foto asli"], horizontal=True,
                            help="Grad-CAM: merah = bagian yang paling memengaruhi keputusan. Fokus daun: area daun "
                                 "bergejala yang dianalisis model (latar diredupkan).")
            if mode == "Grad-CAM":
                st.image(fu.tempel_heatmap(tampil, cam), use_container_width=True,
                         caption="Grad-CAM pada foto asli: merah = paling berpengaruh")
            elif mode == "Fokus daun":
                _, peta, cakupan = fu.terapkan_fokus(np.asarray(Image.fromarray(tampil).resize((IMG_SIZE, IMG_SIZE)),
                                                                dtype=np.float32))
                if cakupan >= 0.02:
                    peta = np.asarray(Image.fromarray(np.uint8(peta * 255)).resize(tampil.shape[1::-1])) / 255.0
                    redup = np.uint8(tampil * (0.3 + 0.7 * peta[..., None]))
                    st.image(redup, use_container_width=True,
                             caption=f"Area daun bergejala yang dianalisis ({cakupan * 100:.0f}% foto); latar diredupkan")
                else:
                    st.image(tampil, use_container_width=True,
                             caption="Tidak ada area gejala yang jelas, jadi seluruh foto dianalisis")
            else:
                st.image(tampil, use_container_width=True, caption="Foto yang dianalisis")
            st.caption(f"Ukuran foto: {foto.size[0]}×{foto.size[1]} piksel (ditampilkan sesuai proporsi aslinya). "
                       f"CNN membaca salinan {IMG_SIZE}×{IMG_SIZE} piksel, sama seperti saat training.")
 
        with kanan:
            if status == "tidak dikenali":
                st.markdown(f"""<div class="hasil-box hasil-ragu"><p class="hasil-label">Hasil deteksi</p>
                    <p class="hasil-nama">Belum dapat dikenali</p>
                    <p class="hasil-latin">Dugaan terdekat: {info['nama']}</p></div>""", unsafe_allow_html=True)
            else:
                kelas_css = "hasil-sehat" if info["sehat"] else "hasil-sakit"
                label = "Hasil deteksi" if status == "yakin" else "Kemungkinan besar"
                st.markdown(f"""<div class="hasil-box {kelas_css}"><p class="hasil-label">{label}</p>
                    <p class="hasil-nama">{info['nama']}</p><p class="hasil-latin">{info['nama_latin']}</p>
                    </div>""", unsafe_allow_html=True)
            st.metric("Tingkat keyakinan model", f"{yakin * 100:.1f}%",
                      help=f"≥ {AMBANG * 100:.0f}%: yakin · {MINIMUM * 100:.0f}–{AMBANG * 100:.0f}%: perlu dicek · "
                           f"< {MINIMUM * 100:.0f}%: belum dapat dikenali")
            if status == "perlu dicek":
                st.warning(f"Keyakinan {yakin * 100:.0f}% (di bawah {AMBANG * 100:.0f}%). Hasil ini **perlu dicek ulang**: "
                           "lihat juga kelas kedua di bawah, atau potong foto agar daun terlihat lebih besar.")
            elif status == "tidak dikenali":
                st.error(f"Keyakinan hanya {yakin * 100:.0f}%. Model belum dapat mengenali foto ini. Pastikan foto "
                         "berisi daun padi, lalu **potong foto** (menu ✂️ di kiri) agar satu daun yang sakit memenuhi "
                         "bingkai, atau gunakan foto yang lebih jelas dan terang.")
            st.write("**Probabilitas setiap kelas**")
            for i in urut:
                st.progress(float(prob[i]), text=f"{cari_info(CLASS_NAMES[i])['nama']} — {prob[i] * 100:.1f}%")
 
        st.divider()
        if status == "tidak dikenali":
            st.info("Informasi penyakit akan ditampilkan setelah foto dapat dikenali.")
        else:
            if status == "perlu dicek":
                st.caption(f"Informasi berikut untuk dugaan **{info['nama']}**; cocokkan dengan ciri-ciri pada daun Anda.")
            if info["sehat"]:
                st.success("Daun terdeteksi sehat. " + info["gejala"])
            else:
                st.subheader("Tentang penyakit ini")
                st.write(f"**Penyebab:** {info['penyebab']}")
                st.write(f"**Gejala:** {info['gejala']}")
            st.write("**Saran penanganan:**")
            for s_ in info["penanganan"]:
                st.markdown(f"- {s_}")
        st.caption("Hasil ini adalah alat bantu deteksi awal, bukan diagnosis pasti. "
                   "Untuk keputusan pengendalian, konsultasikan dengan penyuluh pertanian setempat.")
 
with tab_performa:
    if metrics is None:
        st.info("File `hasil/metrics.json` belum ada. Jalankan training terlebih dahulu.")
    else:
        uji = metrics["uji"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Akurasi uji", f"{uji['akurasi'] * 100:.2f}%")
        k2.metric("Precision (macro)", f"{uji['macro']['precision'] * 100:.2f}%")
        k3.metric("Recall (macro)", f"{uji['macro']['recall'] * 100:.2f}%")
        k4.metric("F1-score (macro)", f"{uji['macro']['f1-score'] * 100:.2f}%")
        ds = metrics["dataset"]
        st.caption(f"Dievaluasi pada {ds['split']['test']} gambar uji yang tidak pernah dilihat model saat training. "
                   f"Total dataset {ds['total']} gambar, {len(ds['kelas'])} kelas.")
 
        st.write("**Kinerja per kelas**")
        st.dataframe([{"Kelas": cari_info(c)["nama"], "Precision": f"{v['precision'] * 100:.1f}%",
                       "Recall": f"{v['recall'] * 100:.1f}%", "F1-score": f"{v['f1-score'] * 100:.1f}%",
                       "Jumlah uji": int(v["support"])} for c, v in uji["per_kelas"].items()],
                     hide_index=True, use_container_width=True)
 
        if metrics.get("skenario"):
            st.write("**Perbandingan skenario**")
            def _aug(a):
                return "Kuat" if a is True else (str(a).capitalize() if a else "Tidak")
 
            def _drop(d):
                if isinstance(d, bool):
                    return "0,5 & 0,3" if d else "Tidak"
                return " & ".join(f"{x:.1f}".replace(".", ",") for x in d if x) or "Tidak"
            ada_pilihan = any(s.get("terpilih") for s in metrics["skenario"])
            st.dataframe([{"Skenario": s["nama"], "Augmentasi": _aug(s["augmentasi"]), "Dropout": _drop(s["dropout"]),
                           "Akurasi validasi": f"{s['val_acc'] * 100:.2f}%",
                           "Akurasi uji": f"{s['test_acc'] * 100:.2f}%",
                           "F1 macro": f"{s['test_f1_macro'] * 100:.2f}%",
                           "Model final": "✅" if (s.get("terpilih") if ada_pilihan else i == len(metrics["skenario"]) - 1)
                           else ""} for i, s in enumerate(metrics["skenario"])],
                         hide_index=True, use_container_width=True)
            pm = metrics.get("pemilihan")
            if pm:
                nama = next((s["nama"] for s in metrics["skenario"] if s["kode"] == pm["terpilih"]), pm["terpilih"])
                st.write(f"**Model yang dipakai website: {nama}**")
                st.caption("Aturan pemilihan: " + pm["aturan"])
                for a_ in pm["alasan"]:
                    st.markdown(f"- {a_}")
 
        fa = metrics.get("fitur_ambang")
        if fa:
            st.write(f"**Fitur ambang keyakinan ({fa['ambang'] * 100:.0f}%)**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Akurasi tanpa ambang", f"{fa['tanpa_ambang']['akurasi'] * 100:.1f}%")
            c2.metric("Akurasi prediksi yang diterima", f"{fa['dengan_ambang']['akurasi'] * 100:.1f}%")
            c3.metric("Porsi gambar diterima", f"{fa['dengan_ambang']['cakupan'] * 100:.1f}%")
 
        for judul, berkas_img in [("Grafik akurasi", "hasil/grafik_akurasi.png"),
                                  ("Grafik loss", "hasil/grafik_loss.png"),
                                  ("Confusion matrix", "hasil/confusion_matrix.png"),
                                  ("Contoh Grad-CAM", "hasil/gradcam_contoh.png")]:
            if os.path.exists(berkas_img):
                st.write(f"**{judul}**")
                st.image(berkas_img, use_container_width=True)
 
with tab_info:
    st.write("Penyakit dan kondisi daun yang dapat dikenali oleh PadiCek:")
    for c in CLASS_NAMES:
        info = cari_info(c)
        with st.expander(f"{info['nama']}  ·  {info['nama_latin']}"):
            if not info["sehat"]:
                st.write(f"**Penyebab:** {info['penyebab']}")
            st.write(f"**Ciri-ciri:** {info['gejala']}")
            st.write("**Penanganan:**")
            for s in info["penanganan"]:
                st.markdown(f"- {s}")

with tab_tentang:
    st.write("PadiCek adalah proyek mata kuliah Pembelajaran Mendalam yang membangun model **Convolutional "
             "Neural Network (CNN)** untuk mengklasifikasikan kondisi daun padi dari foto.")
    st.markdown(
        "- **Model:** CNN klasifikasi multikelas (output softmax, loss categorical cross-entropy, "
        "metrik accuracy) dengan Data Augmentation dan Dropout\n"
        "- **Fitur unik:** Grad-CAM dan ambang keyakinan untuk mendeteksi input di luar domain\n"
        "- **Dataset:** [Rice Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)\n"
        f"- **Kelas:** {', '.join(cari_info(c)['nama'] for c in CLASS_NAMES)}"
    )
    if metrics:
        st.markdown(f"- **Jumlah parameter:** {metrics['model']['total_parameter']:,}".replace(",", ".")
                    + f"\n- **Ukuran input:** {IMG_SIZE}×{IMG_SIZE} piksel")
 