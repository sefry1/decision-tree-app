Decision Tree Streamlit App
===========================

File:
- app.py
- modeling.py
- utils.py
- requirements.txt

Cara menjalankan:
1. Buat virtualenv (opsional) dan install requirement:
   pip install -r requirements.txt

2. Jalankan:
   streamlit run app.py

3. Di sidebar:
   - Upload file Excel (.xlsx) atau CSV, atau biarkan kosong untuk membaca /mnt/data/BlaBla.xlsx
   - Pilih kolom target, atur parameter, lalu tekan "Train Model".

Catatan:
- Jika ingin versi Flask, beri tahu saya.
- Jika datasetmu butuh preprocessing khusus (one-hot encode, scaling, text processing), kasih tahu supaya saya tambahkan pipeline.
