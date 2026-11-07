import numpy as np
import tensorflow as tf
import keras

LAPISAN_KONV_TERAKHIR = "blok4_relu"
AMBANG_DEFAULT = 0.60
AMBANG_MINIMUM = 0.50
FORMAT_GAMBAR = ["jpg", "jpeg", "jfif", "png", "webp", "bmp", "gif", "tif", "tiff"]


def buka_gambar(sumber):
    """Membuka gambar dari file/upload dengan aman untuk berbagai format dari internet/HP:
    rotasi EXIF, PNG/WEBP transparan (ditempel di atas latar putih), PNG 16-bit, CMYK,
    palet warna, dan GIF animasi (diambil frame pertama). Hasil: PIL Image RGB 8-bit."""
    from PIL import Image, ImageOps

    img = Image.open(sumber)
    if getattr(img, "is_animated", False):
        img.seek(0)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):              # 16/32-bit -> skala ke 0-255
        arr = np.asarray(img, dtype=np.float64)
        maks = arr.max() if arr.max() > 0 else 1.0
        img = Image.fromarray(np.uint8(arr / maks * 255)).convert("L")
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    if img.mode in ("RGBA", "LA"):
        latar = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(latar, img.convert("RGBA"))
    return img.convert("RGB")

def peta_daun(x, ukuran_peta=112):
    """Peta area daun bergejala (0-1) dari batch gambar x (B, H, W, 3) bernilai 0-255.
    1) Warna gejala (kuning, jingga, coklat, kemerahan) dicari di ruang warna HSV.
    2) Opening (erosi lalu dilasi 3x3) membuang bintik kecil/noise.
    3) Dilasi 9x9 menyatukan bercak-bercak menjadi satu area daun.
    4) Konvolusi rata-rata 7x7 menghaluskan tepi peta.
    Erosi = -maxpool(-x) dan dilasi = maxpool(x): operasi morfologi dalam bentuk konvolusi."""
    x = tf.convert_to_tensor(x, tf.float32)
    tinggi, lebar = tf.shape(x)[1], tf.shape(x)[2]
    kecil = tf.clip_by_value(tf.image.resize(x, (ukuran_peta, ukuran_peta)) / 255.0, 0.0, 1.0)
    hsv = tf.image.rgb_to_hsv(kecil)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gejala = ((h < 50 / 360) | (h > 340 / 360)) & (s > 0.22) & (v > 0.20)
    m = tf.cast(gejala, tf.float32)[..., None]
    m = -tf.nn.max_pool2d(-m, 3, 1, "SAME")          # erosi
    m = tf.nn.max_pool2d(m, 3, 1, "SAME")            # dilasi (opening selesai)
    m = tf.nn.max_pool2d(m, 13, 1, "SAME")           # dilasi besar: satukan bercak jadi area daun
    m = tf.nn.avg_pool2d(m, 7, 1, "SAME")            # haluskan tepi
    return tf.image.resize(m, (tinggi, lebar))


@keras.saving.register_keras_serializable(package="PadiCek")
class FokusDaun(keras.layers.Layer):
    """Lapisan preprocessing di awal model: area daun bergejala tetap tajam, latar belakang dikaburkan.
    Jika area gejala < min_cakupan (misalnya daun sehat), gambar dibiarkan utuh."""

    def __init__(self, min_cakupan=0.02, **kwargs):
        super().__init__(**kwargs)
        self.min_cakupan = min_cakupan

    def call(self, x):
        m = peta_daun(x)
        cakupan = tf.reduce_mean(tf.cast(m > 0.5, tf.float32), axis=[1, 2, 3], keepdims=True)
        m = tf.where(cakupan >= self.min_cakupan, m, tf.ones_like(m))
        kabur = tf.nn.avg_pool2d(tf.nn.avg_pool2d(x, 11, 1, "SAME"), 11, 1, "SAME")
        return m * x + (1.0 - m) * kabur

    def get_config(self):
        return {**super().get_config(), "min_cakupan": self.min_cakupan}


def terapkan_fokus(gambar, min_cakupan=0.02):
    """Versi numpy untuk tampilan: (hasil fokus daun, peta daun 0-1, cakupan)."""
    g = np.asarray(gambar, dtype=np.float32)[None]
    m = peta_daun(g)[0, ..., 0].numpy()
    hasil = FokusDaun(min_cakupan)(tf.constant(g))[0].numpy()
    return hasil, m, float((m > 0.5).mean())


def fokus_gradcam_pada_daun(model_gradcam, gambar_list, min_cakupan=0.02):
    """Mengukur bug 'Grad-CAM menyorot latar': porsi energi Grad-CAM yang jatuh di area daun bergejala
    dibanding luas area tersebut. Rasio > 1 berarti model lebih memperhatikan daun daripada latar."""
    from PIL import Image
    energi, luas = [], []
    for g in gambar_list:
        g = np.asarray(g, dtype=np.float32)
        m = peta_daun(g[None])[0, ..., 0].numpy() > 0.5
        if m.mean() < min_cakupan:
            continue
        cam, _ = gradcam(model_gradcam, g)
        cam = np.asarray(Image.fromarray(np.float32(cam)).resize(g.shape[1::-1], Image.BILINEAR))
        energi.append(float((cam * m).sum() / (cam.sum() + 1e-8))); luas.append(float(m.mean()))
    if not energi:
        return {"n": 0, "energi_di_daun": 0.0, "luas_daun": 0.0, "rasio": 0.0, "porsi_gambar_fokus": 0.0}
    e, l = np.array(energi), np.array(luas)
    return {"n": int(len(e)), "energi_di_daun": float(e.mean()), "luas_daun": float(l.mean()),
            "rasio": float(e.mean() / (l.mean() + 1e-8)), "porsi_gambar_fokus": float((e > l).mean())}


def siapkan_gambar(gambar, ukuran):
    """Resize PERSIS seperti pipeline training (tf.image.resize bilinear), lalu kembalikan array
    float32 bernilai 0-255. Dipakai website, Grad-CAM, dan uji luar domain agar tidak ada
    perbedaan preprocessing antara training dan pemakaian (train-serving skew)."""
    import tensorflow as tf

    arr = np.asarray(gambar.convert("RGB") if hasattr(gambar, "convert") else gambar, dtype=np.float32)
    return tf.image.resize(arr, (ukuran, ukuran)).numpy()


def buat_model_gradcam(model):
    """Model bantu: input gambar -> (feature map konvolusi terakhir, probabilitas kelas)."""
    import tensorflow as tf

    return tf.keras.Model(model.inputs,
                          [model.get_layer(LAPISAN_KONV_TERAKHIR).output, model.output])


def gradcam(model_gradcam, gambar, kelas=None):
    """Menghitung heatmap Grad-CAM (Selvaraju dkk., 2017).

    gambar : array (H, W, 3) bernilai 0-255 (tanpa normalisasi, rescaling ada di dalam model)
    kelas  : indeks kelas yang dijelaskan; None = kelas dengan probabilitas tertinggi
    return : (heatmap ukuran feature map bernilai 0-1, vektor probabilitas)
    """
    import tensorflow as tf

    x = tf.convert_to_tensor(np.asarray(gambar, dtype=np.float32)[None, ...])
    with tf.GradientTape() as tape:
        fitur, prob = model_gradcam(x, training=False)
        if kelas is None:
            kelas = int(tf.argmax(prob[0]))
        skor = prob[:, kelas]
    grad = tape.gradient(skor, fitur)                    # d skor / d feature map
    bobot = tf.reduce_mean(grad, axis=(1, 2))            # rata-rata gradien per filter
    cam = tf.nn.relu(tf.reduce_sum(fitur * bobot[:, None, None, :], axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), prob[0].numpy()

def tempel_heatmap(gambar, cam, alpha=0.45):
    """Menimpa heatmap (colormap 'jet') di atas gambar asli. Mengembalikan array uint8."""
    from matplotlib import colormaps
    from PIL import Image

    gambar = np.clip(np.asarray(gambar), 0, 255).astype(np.uint8)
    h, w = gambar.shape[:2]
    cam_besar = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)) / 255.0
    warna = colormaps["jet"](cam_besar)[..., :3] * 255
    return np.uint8(gambar * (1 - alpha) + warna * alpha)

def status_prediksi(prob, ambang=AMBANG_DEFAULT, minimum=AMBANG_MINIMUM):
    """Fitur 2: tiga tingkat keyakinan.
    - 'yakin'           : keyakinan >= ambang (dipilih dari data validasi) -> hasil ditampilkan penuh
    - 'perlu dicek'     : minimum <= keyakinan < ambang -> dugaan tetap ditampilkan + peringatan
    - 'tidak dikenali'  : keyakinan < minimum (kelas teratas tidak lebih besar dari gabungan kelas lain)
                          -> kemungkinan bukan daun padi / foto kurang jelas, pengguna diminta foto ulang"""
    keyakinan = float(np.max(prob))
    if keyakinan >= ambang:
        return "yakin", keyakinan
    return ("perlu dicek" if keyakinan >= min(minimum, ambang) else "tidak dikenali"), keyakinan


def pilih_ambang(prob_val, y_val, kandidat=None, target_akurasi=0.95):
    """Memilih ambang terkecil yang membuat akurasi prediksi yang diterima pada data VALIDASI
    mencapai target. Data uji tidak dipakai agar evaluasi tetap jujur."""
    kandidat = kandidat or [round(0.50 + 0.05 * i, 2) for i in range(10)]   # 0,50 ... 0,95
    keyakinan = prob_val.max(axis=1)
    benar = prob_val.argmax(axis=1) == y_val
    for t in kandidat:
        diterima = keyakinan >= t
        if diterima.sum() > 0 and benar[diterima].mean() >= target_akurasi:
            return float(t)
    return float(kandidat[-1])


def tabel_ambang(prob, y, kandidat):
    """Cakupan (porsi prediksi yang diterima) dan akurasi prediksi yang diterima untuk tiap ambang."""
    keyakinan = prob.max(axis=1)
    benar = prob.argmax(axis=1) == y
    baris = []
    for t in kandidat:
        d = keyakinan >= t
        baris.append({"ambang": float(t), "cakupan": float(d.mean()),
                      "akurasi": float(benar[d].mean()) if d.sum() else 0.0, "n_diterima": int(d.sum())})
    return baris


def buat_input_luar_domain(contoh_gambar, ukuran, n_per_jenis=40, seed=42):
    """Membuat gambar uji 'di luar domain' untuk menguji fitur 2:
    - noise acak, - warna polos/gradien, - potongan daun yang diacak (warna mirip, struktur hilang)."""
    rng = np.random.default_rng(seed)
    hasil = {"Noise acak": [], "Warna polos & gradien": [], "Potongan citra diacak": []}
    for _ in range(n_per_jenis):
        hasil["Noise acak"].append(rng.integers(0, 256, (ukuran, ukuran, 3)).astype(np.float32))
        warna_a, warna_b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        t = np.linspace(0, 1, ukuran)[None, :, None] if rng.random() < 0.5 else np.zeros((1, 1, 1))
        hasil["Warna polos & gradien"].append((warna_a * (1 - t) + warna_b * t + np.zeros((ukuran, ukuran, 3)))
                                              .astype(np.float32))
        g = np.asarray(contoh_gambar[rng.integers(len(contoh_gambar))], dtype=np.float32)
        k = ukuran // 8
        ubin = [g[i:i + k, j:j + k] for i in range(0, k * 8, k) for j in range(0, k * 8, k)]
        rng.shuffle(ubin)
        acak = np.zeros_like(g)
        for idx, u in enumerate(ubin):
            i, j = divmod(idx, 8)
            acak[i * k:(i + 1) * k, j * k:(j + 1) * k] = u
        hasil["Potongan citra diacak"].append(acak)
    return {k: np.stack(v) for k, v in hasil.items()}


def uji_penghapusan_gradcam(model_gradcam, model, gambar_list, porsi=0.2, seed=42):
    """Bukti dampak Grad-CAM (deletion test): menutup 20% area yang paling 'merah' menurut
    Grad-CAM lalu membandingkan penurunan keyakinan dengan menutup 20% area acak.
    Jika Grad-CAM benar menunjuk area penting, penurunannya harus lebih besar."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    turun_cam, turun_acak = [], []
    for g in gambar_list:
        g = np.asarray(g, dtype=np.float32)
        cam, prob = gradcam(model_gradcam, g)
        kelas = int(prob.argmax())
        n = max(1, int(round(cam.size * porsi)))
        topk = np.zeros(cam.size, bool); topk[np.argsort(cam.ravel())[-n:]] = True
        acak = np.zeros(cam.size, bool); acak[rng.choice(cam.size, n, replace=False)] = True
        hasil = []
        for m in (topk, acak):
            topeng = np.asarray(Image.fromarray(np.uint8(m.reshape(cam.shape)) * 255)
                                .resize(g.shape[1::-1], Image.NEAREST)) > 0
            tertutup = g.copy(); tertutup[topeng] = 127.0
            hasil.append(float(model.predict(tertutup[None], verbose=0)[0][kelas]))
        turun_cam.append(prob[kelas] - hasil[0]); turun_acak.append(prob[kelas] - hasil[1])
    return {"n": len(gambar_list), "porsi_area": porsi,
            "penurunan_gradcam": float(np.mean(turun_cam)), "penurunan_acak": float(np.mean(turun_acak)),
            "persen_lebih_besar": float(np.mean(np.array(turun_cam) > np.array(turun_acak)))}
