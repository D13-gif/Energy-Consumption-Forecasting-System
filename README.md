# EnergiCast — Energy Consumption Forecasting System

![EnergiCast](https://img.shields.io/badge/version-1.0-blue)
![Python](https://img.shields.io/badge/python-3.11%20(ML)%20%7C%203.12%20(Vercel)-blue)
![Flask](https://img.shields.io/badge/flask-2.3%2B-green)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Aplikasi web untuk forecasting konsumsi listrik rumah tangga menggunakan model LSTM (Long Short-Term Memory). Dibangun dengan Flask, TensorFlow/Keras, AdminLTE 4, dan Chart.js.

## 🎯 Tujuan Proyek

- Mengimplementasikan workflow forecasting time-series menggunakan LSTM end-to-end
- Menyajikan eksplorasi data, preprocessing, training, evaluasi, dan forecasting dalam web UI interaktif
- Menjadikan proyek mudah dijalankan secara lokal, dipelajari, dan dipresentasikan sebagai portofolio
- Menyediakan struktur kode modular, dokumentasi, pengujian, serta reproducible workflow

## ✨ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **Dashboard** | Overview KPI, grafik historis, ringkasan forecast, status model |
| **Data Explorer** | Preview dataset dengan filter tanggal, pagination, validasi missing values |
| **Preprocessing** | Pipeline: parse timestamp, handle missing, agregasi hourly, scaling, split kronologis, windowing |
| **EDA** | Statistik deskriptif, time series, profil jam/hari, distribusi target, missing values |
| **Model Training** | Konfigurasi LSTM, training dengan early stopping, progress tracking, artifact saving |
| **Model Performance** | MAE/RMSE/MSE, actual vs predicted, residual distribution, loss curves |
| **Forecast** | Multi-step recursive forecast, actual/predicted/forecast chart, export CSV |
| **Model Information** | Arsitektur, hyperparameter, I/O, preprocessing config, metadata, penjelasan LSTM |

## 🛠 Tech Stack

**Backend:**
- Python 3.11 (full ML) / Python 3.12 (Vercel dashboard)
- Flask 2.3+ (App Factory Pattern, Blueprints)
- SQLAlchemy 2.0+ (SQLite)
- TensorFlow/Keras 2.13+ (LSTM)
- Scikit-learn (Preprocessing, Metrics)
- Pandas, NumPy (Data Processing)

**Frontend:**
- AdminLTE 4 (HTML, CSS, Vanilla JS)
- Bootstrap 5
- Chart.js 4 (Interactive Charts)
- Font Awesome 6 (Icons)
- Jinja2 Templates

## 📁 Struktur Proyek

```
energicast/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── config.py            # Konfigurasi (dev/prod/test)
│   ├── extensions.py        # Ekstensi (SQLAlchemy)
│   ├── routes/              # Blueprint routes
│   │   ├── dashboard.py
│   │   ├── data.py
│   │   ├── preprocessing.py
│   │   ├── eda.py
│   │   ├── training.py
│   │   ├── forecast.py
│   │   ├── performance.py
│   │   └── info.py
│   ├── services/            # Business logic layer
│   ├── repositories/        # Data access layer
│   ├── utils/               # Utilities (validators, logging, time)
│   ├── templates/           # Jinja2 templates
│   └── static/              # CSS, JS, images
├── ml/                      # ML pipeline modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── windowing.py
│   ├── model_builder.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── forecaster.py
├── data/
│   ├── raw/                 # Dataset mentah (not committed)
│   └── processed/           # Data hasil preprocessing (not committed)
├── artifacts/
│   ├── models/              # Model artifacts (.keras)
│   ├── scalers/             # Scaler artifacts (.pkl)
│   ├── plots/               # Generated plots
│   └── metadata/            # Metadata JSON
├── instance/                # SQLite database (not committed)
├── tests/                   # Unit & integration tests
├── public/static/           # Vercel CDN assets
├── scripts/                 # CLI scripts
├── docs/                    # Documentation
├── requirements.txt
├── requirements-ml.txt
├── requirements-dev.txt
├── vercel.json
├── main.py
├── run.py
├── .env.example
└── README.md
```

## 🚀 Instalasi & Menjalankan

### Prasyarat
- Python 3.11 untuk aplikasi lokal lengkap (TensorFlow 2.13)
- Python 3.12 untuk mode dashboard Vercel
- pip (Python package manager)

### Langkah-langkah

1. **Clone/Extract proyek dan masuk ke direktori:**
   ```bash
   cd energicast
   ```

2. **Buat virtual environment (direkomendasikan):**
   ```bash
   # Windows: Python 3.11 untuk aplikasi ML lengkap
   py -3.11 -m venv venv
   venv\Scripts\activate
   # Linux/Mac:
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   Untuk aplikasi lokal lengkap, gunakan Python 3.11:
   ```bash
   pip install -r requirements.txt -r requirements-ml.txt
   ```
   Dependensi Vercel sengaja hanya memuat Flask dan Flask-SQLAlchemy; TensorFlow tidak dipasang pada runtime tersebut.

4. **Siapkan dataset:**
   - Download dataset "Individual Household Electric Power Consumption" dari [Kaggle](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) atau [UCI](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
   - Letakkan file `household_power_consumption.txt` di folder `energicast/data/raw/`
   - Dataset tidak disimpan di Git karena ukurannya besar. File lokal yang ada di `data/raw/household_power_consumption.txt` diabaikan oleh `.gitignore`.

5. **Konfigurasi environment (opsional):**
   ```bash
   copy .env.example .env
   # Edit .env sesuai kebutuhan
   ```

6. **Jalankan aplikasi:**
   ```bash
   python run.py
   ```

7. **Buka browser:**
   ```
   http://localhost:5000
   ```

## 📖 Alur Penggunaan

1. **Dashboard** - Lihat status sistem dan overview
2. **Data Explorer** - Validasi dataset, lihat missing values, eksplorasi data
3. **Preprocessing** - Konfigurasi dan jalankan pipeline preprocessing
4. **EDA** - Analisis statistik dan visualisasi data
5. **Model Training** - Konfigurasi hyperparameter, latih model LSTM
6. **Model Performance** - Evaluasi metrik (MAE, RMSE, MSE), lihat grafik actual vs predicted
7. **Forecast** - Generate prediksi masa depan, export ke CSV
8. **Model Information** - Lihat detail arsitektur, hyperparameter, metadata

## ⚙️ Konfigurasi Default

| Parameter | Nilai Default | Deskripsi |
|-----------|---------------|-----------|
| Agregasi | Hourly (1 jam) | Rata-rata per jam dari data per menit |
| Window Size | 24 | 24 timestep historis (24 jam) |
| Train/Val/Test | 70/15/15 | Split kronologis |
| LSTM Units | 32 | 1 LSTM layer |
| Dropout | 0.1 | Regularisasi |
| Epochs | 20 | Max epochs (early stopping) |
| Batch Size | 32 | Training batch size |
| Learning Rate | 0.001 | Adam optimizer |
| Loss | MSE | Mean Squared Error |
| Forecast Horizons | 24, 48, 72, 168 jam | Horizon yang didukung |

## 🧪 Testing

```bash
# Instal dependensi test
pip install -r requirements-dev.txt

# Jalankan smoke test dashboard dan mode read-only
python -m pytest -q
```

Test CI mencakup halaman, aset statis, health check, dan penolakan endpoint API yang tidak tersedia pada Vercel. Test tersebut tidak menjalankan training TensorFlow.

## 🚀 Deployment

- [Panduan GitHub Actions dan Vercel](docs/deployment.md)
- Vercel melayani dashboard Flask dalam mode read-only; deployment ini tidak menyertakan dataset maupun model dan tidak menjalankan training/inference.
- Untuk pipeline ML penuh, jalankan aplikasi di lingkungan kontainer persisten sesuai panduan deployment.

## 📊 API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/api/health` | Health check |
| GET | `/api/data/summary` | Ringkasan dataset |
| GET | `/api/data/preview` | Preview data (pagination) |
| GET | `/api/data/validation` | Validasi & missing values |
| GET | `/api/eda/overview` | Statistik deskriptif |
| GET | `/api/eda/timeseries` | Data time series |
| GET | `/api/eda/hourly-profile` | Profil rata-rata per jam |
| GET | `/api/eda/weekly-profile` | Profil rata-rata per hari |
| GET | `/api/eda/distribution` | Distribusi target |
| GET | `/api/eda/missing-summary` | Ringkasan missing values |
| POST | `/api/preprocessing/run` | Jalankan preprocessing |
| GET | `/api/preprocessing/status` | Status preprocessing |
| POST | `/api/training/start` | Mulai training |
| GET | `/api/training/status/<run_id>` | Status training |
| GET | `/api/model/metrics` | Metrik evaluasi |
| GET | `/api/model/history` | Training history |
| GET | `/api/model/info` | Info model |
| POST | `/api/forecast` | Generate forecast |
| GET | `/api/forecast/latest` | Forecast terbaru |

## 📝 Catatan Penting

### Batasan Model
- **Univariate baseline:** Hanya menggunakan `Global_active_power` historis
- **Recursive forecasting:** Error accumulation pada horizon panjang
- **No exogenous variables:** Cuaca, hari libur, musim tidak dimodelkan
- **Single household:** Dataset hanya satu rumah tangga
- **Demo/learning purpose:** Bukan untuk keputusan operasional nyata

### Time-Series Integrity
- ✅ No random shuffle
- ✅ Chronological train/val/test split
- ✅ Scaler fit only on training data
- ✅ No future leakage in windowing
- ✅ Metrics calculated on held-out test set (inverse transformed)

## 📄 Lisensi

MIT License - Lihat file [LICENSE](LICENSE) untuk detail.

## 🙏 Acknowledgments

- **AdminLTE 4** - Dashboard template
- **Chart.js** - Charting library
- **TensorFlow/Keras** - Deep learning framework
- **Dataset:** Individual Household Electric Power Consumption (UCI/Kaggle)
- **Font Awesome** - Icons

## 📞 Kontak & Kontribusi

Proyek ini dibuat untuk tujuan akademik (Tugas Akhir). Kontribusi, saran, dan laporan bug sangat diterima.

---
*EnergiCast v1.0 — Energy Consumption Forecasting System*