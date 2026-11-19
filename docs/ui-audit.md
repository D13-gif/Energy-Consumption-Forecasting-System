# Audit dan perbaikan UI EnergiCast

## Temuan

- Aset yang dipakai adalah AdminLTE 4.10.2. CSS AdminLTE memuat Bootstrap 5.3.9, sedangkan markup `base.html` sebelumnya masih memakai kerangka AdminLTE 3 (`.wrapper`, `.main-header`, `.main-sidebar`, `.content-wrapper`, `.main-footer`, dan `.control-sidebar`) serta atribut `data-widget`. Hasil ukur browser sebelum perbaikan: sidebar `position: static`, lebarnya 1424.8 px pada viewport 1440 px, dan wrapper tetap `display: block`; CSS v4 tidak mempunyai selector layout v3 tersebut.
- Beberapa utilitas Bootstrap 4 juga tertinggal (`font-weight-bold`, `float-sm-right`). Sidebar ringkas memakai class buatan yang tidak dikenal AdminLTE 4. Semua halaman utama sudah mewarisi `base.html`, sehingga satu perbaikan kerangka memperbaiki layout bersama.
- Urutan stylesheet sudah benar: CSS AdminLTE, palet warna opt-in, ikon, lalu CSS aplikasi. Aset vendor pada proyek cocok dengan sumber AdminLTE 4.10.2. Bootstrap JavaScript sebelumnya 5.3.2, tidak sama dengan Bootstrap CSS bundle 5.3.9.
- Browser memilih mode gelap OS karena elemen `<html>` tidak menetapkan `data-bs-theme`. Pengaturan gelap juga masih memakai `.dark-mode` AdminLTE 3, yang tidak mengubah tema Bootstrap/AdminLTE 4.
- Chart.js termuat dan tidak ada error console/network pada pemuatan dashboard. Endpoint historis mengembalikan `success: false` dengan pesan `Preprocessing artifacts not found`; karena itu chart memang belum memiliki data olahan. Ruang chart kosong tetap 300 px saat canvas disembunyikan.
- Kotak abu-abu tepat mengelilingi teks pada screenshot adalah tampilan seleksi teks browser; snapshot browser pemeriksaan tidak memiliki seleksi aktif. Ini bukan background CSS pada label.

## Perubahan

- Migrasi kerangka bersama ke `.app-wrapper`, `.app-header`, `.app-sidebar`, `.app-main`, dan `.app-footer`; menu memakai `sidebar-menu` dan `data-lte-toggle` sesuai AdminLTE 4.
- Memperbaiki layout fixed/responsif, breadcrumb Bootstrap 5, active state/ARIA navigasi, footer, dan menempatkan pengaturan tema/sidebar di dropdown navbar. Kontrol tema dan sidebar ringkas tetap tersedia.
- Menetapkan tema terang sebagai default, sidebar navy gelap, serta menghubungkan toggle ke `data-bs-theme`, `sidebar-mini`, dan `sidebar-collapse`.
- Menata ulang spacing, kartu, tipografi, chart, loading/empty state, dan breakpoint mobile. Saat data olahan belum ada, area chart disembunyikan dan pesan Bahasa Indonesia beserta langkah berikutnya ditampilkan. Tidak ada data sintetis yang ditambahkan.
- Dashboard kini meminta data historis pada pemuatan awal (rentang 7 hari), bukan hanya setelah tombol rentang diklik. Sidebar mobile diuji dapat dibuka dan ditutup melalui overlay; preferensi sidebar ringkas tidak lagi menimpa perilaku responsif AdminLTE.
- Menyamakan Bootstrap JavaScript dengan Bootstrap CSS 5.3.9 dan memuat Source Sans 3 yang sebelumnya hanya disebut sebagai font fallback.

## Batas pengujian

UI diuji secara lokal pada desktop dan viewport mobile menggunakan browser. Grafik data tidak bisa diuji dengan nilai karena preprocessing artifact belum tersedia; setelah menjalankan preprocessing dari halaman aplikasi, grafik dapat diverifikasi dengan dataset asli.
