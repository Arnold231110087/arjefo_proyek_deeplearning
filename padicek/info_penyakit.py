import re

INFO = {
    "bacterial_leaf_blight": {
        "nama": "Hawar Daun Bakteri (Kresek)",
        "nama_latin": "Bacterial Leaf Blight",
        "penyebab": "Bakteri Xanthomonas oryzae pv. oryzae",
        "gejala": (
            "Bercak kebasahan berwarna hijau keabu-abuan dimulai dari ujung atau tepi daun, "
            "kemudian meluas menjadi garis kuning hingga putih keabu-abuan dengan tepi bergelombang. "
            "Pada serangan berat daun mengering seperti terbakar."
        ),
        "penanganan": [
            "Gunakan varietas tahan dan benih sehat bersertifikat.",
            "Hindari pemupukan nitrogen (urea) berlebihan; seimbangkan dengan kalium.",
            "Atur pengairan berselang (intermittent) dan jaga sanitasi pematang.",
            "Konsultasikan pengendalian kimia dengan penyuluh pertanian setempat bila serangan meluas.",
        ],
        "sehat": False,
    },
    "brown_spot": {
        "nama": "Bercak Coklat",
        "nama_latin": "Brown Spot",
        "penyebab": "Jamur Bipolaris oryzae (Cochliobolus miyabeanus)",
        "gejala": (
            "Bercak oval berwarna coklat dengan bagian tengah keabu-abuan dan tepi coklat kemerahan, "
            "tersebar merata di permukaan daun. Sering muncul pada tanah yang miskin hara."
        ),
        "penanganan": [
            "Perbaiki kesuburan tanah dengan pemupukan berimbang (terutama kalium dan silika).",
            "Gunakan benih sehat dan lakukan perlakuan benih sebelum tanam.",
            "Bersihkan sisa tanaman terinfeksi setelah panen.",
            "Gunakan fungisida sesuai anjuran penyuluh bila serangan berat.",
        ],
        "sehat": False,
    },
    "healthy_rice_leaf": {
        "nama": "Daun Sehat",
        "nama_latin": "Healthy Rice Leaf",
        "penyebab": "-",
        "gejala": "Daun berwarna hijau merata tanpa bercak, garis, atau bagian yang mengering.",
        "penanganan": [
            "Pertahankan pemupukan berimbang dan pengairan yang baik.",
            "Lakukan pemantauan rutin minimal sekali seminggu untuk deteksi dini.",
        ],
        "sehat": True,
    },
    "leaf_blast": {
        "nama": "Blas Daun",
        "nama_latin": "Leaf Blast",
        "penyebab": "Jamur Pyricularia oryzae (Magnaporthe oryzae)",
        "gejala": (
            "Bercak berbentuk belah ketupat (seperti mata) dengan bagian tengah putih keabu-abuan "
            "dan tepi coklat. Bercak dapat menyatu sehingga daun mengering."
        ),
        "penanganan": [
            "Tanam varietas tahan blas dan lakukan pergiliran varietas.",
            "Hindari pupuk nitrogen berlebihan dan jarak tanam terlalu rapat.",
            "Musnahkan jerami dan sisa tanaman yang terinfeksi.",
            "Aplikasikan fungisida sesuai anjuran penyuluh pada awal gejala.",
        ],
        "sehat": False,
    },
    "leaf_scald": {
        "nama": "Hangus Daun (Leaf Scald)",
        "nama_latin": "Leaf Scald",
        "penyebab": "Jamur Microdochium oryzae (Monographella albescens)",
        "gejala": (
            "Bercak memanjang mulai dari ujung atau tepi daun dengan pola pita bergelombang "
            "berwarna coklat muda hingga coklat kemerahan, sehingga ujung daun tampak hangus."
        ),
        "penanganan": [
            "Gunakan benih sehat dan varietas yang lebih tahan.",
            "Kurangi pemupukan nitrogen berlebihan.",
            "Jaga jarak tanam agar sirkulasi udara baik dan kelembapan tidak terlalu tinggi.",
            "Konsultasikan penggunaan fungisida dengan penyuluh pertanian.",
        ],
        "sehat": False,
    },
    "sheath_blight": {
        "nama": "Hawar Pelepah",
        "nama_latin": "Sheath Blight",
        "penyebab": "Jamur Rhizoctonia solani",
        "gejala": (
            "Bercak oval kehijauan hingga abu-abu dengan tepi coklat pada pelepah dan daun, "
            "biasanya dimulai dekat permukaan air lalu menjalar ke atas."
        ),
        "penanganan": [
            "Atur jarak tanam tidak terlalu rapat (misalnya sistem jajar legowo).",
            "Hindari pemupukan nitrogen berlebihan.",
            "Keringkan petak sawah secara berkala untuk menurunkan kelembapan.",
            "Gunakan fungisida atau agens hayati (misalnya Trichoderma) sesuai anjuran penyuluh.",
        ],
        "sehat": False,
    },
}

_KATA_KUNCI = {
    "bacterial_leaf_blight": ["bacterial", "blight_bact", "bacterialleafblight", "blb"],
    "brown_spot": ["brown"],
    "healthy_rice_leaf": ["healthy", "sehat", "normal"],
    "leaf_blast": ["blast"],
    "leaf_scald": ["scald"],
    "sheath_blight": ["sheath"],
}


def _normal(teks: str) -> str:
    return re.sub(r"[^a-z0-9]", "", teks.lower())


def cari_info(nama_kelas: str) -> dict:
    """Mengembalikan info penyakit untuk nama folder kelas apa pun dari dataset."""
    n = _normal(nama_kelas)
    for kunci, daftar in _KATA_KUNCI.items():
        if any(_normal(k) in n for k in daftar):
            return INFO[kunci]

    return {
        "nama": nama_kelas.replace("_", " ").title(),
        "nama_latin": nama_kelas,
        "penyebab": "-",
        "gejala": "-",
        "penanganan": ["Konsultasikan dengan penyuluh pertanian setempat."],
        "sehat": False,
    }


def nama_indonesia(nama_kelas: str) -> str:
    return cari_info(nama_kelas)["nama"]
