# GitHub Actions dan deployment

## GitHub Actions

Workflow `.github/workflows/ci.yml` menjalankan smoke test pada Python 3.12 untuk setiap pull request dan push ke `main`. CI menginstal dependensi web dan test saja; TensorFlow dan dataset tidak dibutuhkan untuk memvalidasi mode dashboard Vercel.

## Vercel: dashboard read-only

1. Import repository GitHub ke Vercel dan gunakan root repository sebagai Root Directory.
2. Biarkan Framework Preset terdeteksi sebagai Flask/Python dan gunakan konfigurasi dari `vercel.json`.
3. Deploy. Tidak ada start command, secret, maupun environment variable wajib untuk mode dashboard.
4. Verifikasi `/`, navigasi, aset di `/static/`, serta `/api/health`. Health response secara eksplisit menandai dataset, model, training, dan inference tidak tersedia.

Vercel sengaja tidak menerima `requirements-ml.txt` sebagai daftar dependensinya. `main.py` memuat aplikasi Flask terpisah yang tidak mengimpor layanan ML. Endpoint API analitik dan mutasi mengembalikan HTTP 503; UI menampilkan banner read-only dan tidak mengarang statistik atau prediksi. Aset yang dipakai mode ini disalin di `public/static/`; jika aset `app/static/` berubah, salin ulang file yang sama ke lokasi tersebut.

### Batas kompatibilitas

Dokumentasi resmi yang diperiksa pada 19 November 2026:

- [Flask di Vercel](https://vercel.com/docs/frameworks/backend/flask): aplikasi WSGI dijalankan sebagai Vercel Function; aset statis sebaiknya ada di `public/**`.
- [Python runtime](https://vercel.com/docs/functions/runtimes/python): versi yang didukung saat ini adalah Python 3.12–3.14. Bundle Python standar maksimal 500 MB.
- [Batas Function](https://vercel.com/docs/functions/limitations): memori Hobby 2 GB; durasi maksimum standar Hobby 300 detik. Dependency ML berbiner besar dapat menghabiskan batas bundle.
- [Filesystem runtime](https://vercel.com/docs/functions/runtimes): filesystem read-only kecuali `/tmp` maksimal 500 MB; isi `/tmp` khusus instance dan tidak persisten.
- [Penyimpanan Vercel](https://vercel.com/docs/storage): data persisten memerlukan storage/database eksternal.
- [TensorFlow versi yang diuji](https://www.tensorflow.org/install/source#tested_build_configurations): TensorFlow 2.13.0 diuji pada Python 3.8–3.11, bukan versi runtime Vercel yang didukung saat ini.

Karena versi TensorFlow proyek tidak kompatibel dengan Python Vercel yang didukung, bundlenya besar, dan aplikasi lokal menulis SQLite serta model ke disk, deployment Vercel ini hanya untuk tampilan dashboard dan health check. Jangan aktifkan training atau inference pada mode tersebut.

## Alternatif untuk ML penuh

Jalankan `run.py` sebagai web service kontainer pada host yang menyediakan disk persisten dan proses web yang terus hidup (misalnya Render paid web service atau VM/container host sejenis):

1. Gunakan Python 3.11 dan install `requirements.txt` serta `requirements-ml.txt`.
2. Pasang dataset sebagai file terkelola di `data/raw/`; jangan masukkan file dataset 133 MB ke Git.
3. Mount persistent volume, misalnya di `/var/data`, lalu set `DATA_DIR`, `ARTIFACTS_DIR`, dan `INSTANCE_DIR` melalui konfigurasi host; simpan dataset di `data/raw/` pada volume tersebut. `SQLITE_DB_PATH` dapat diarahkan eksplisit ke file database di volume. Alternatifnya, migrasikan metadata ke database terkelola dan model ke object storage.
4. Set `FLASK_CONFIG=production` dan `SECRET_KEY` melalui secret manager host; arahkan health check ke `/health`.
5. Tetap jalankan CI sebelum deploy. Training sekarang berjalan dalam proses aplikasi; untuk pemakaian multi-user/produksi, pindahkan training ke worker antrean terpisah sebelum menerima workload tersebut.

Alternatif ini mempertahankan runtime Python yang cocok untuk TensorFlow dan menyimpan database/model di luar filesystem sementara. Tidak ada deployment Vercel atau host backend yang dibuat oleh repository ini secara otomatis.
