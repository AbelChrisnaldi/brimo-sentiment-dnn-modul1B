from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
HERE=Path(__file__).resolve().parent
st.set_page_config(page_title='Analisis Sentimen BRImo - DNN',page_icon='💬')
@st.cache_resource
def get_model(): return joblib.load(HERE/'model_sentimen_brimo.pkl')
@st.cache_data
def get_info(): return json.loads((HERE/'model_info.json').read_text(encoding='utf-8'))
if not (HERE/'model_sentimen_brimo.pkl').exists() or not (HERE/'model_info.json').exists():
    st.error('Model atau informasi model belum tersedia. Jalankan Modul_1A/train_model.py lalu salin model_sentimen_brimo.pkl dan model_info.json ke folder Modul_1B.')
    st.stop()
model=get_model();info=get_info()
st.title('💬 Klasifikasi Sentimen Ulasan BRImo (DNN)')
st.caption('Modul 1B · model library scikit-learn · teks ulasan, bukan rating bintang')
with st.sidebar:
    st.subheader('Model')
    st.write(info['nama_model'])
    st.metric('Accuracy pada test utama',f"{info['metrik']['accuracy']:.1%}")
    st.metric('F1 Macro pada test utama',f"{info['metrik']['f1_macro']:.3f}")
    st.caption(f"Uji pada {info['metrik']['n_test']} baris")
@st.cache_data

def contoh_csv():
    """Buat sampel unduhan dari CSV BRImo di folder modul ini."""
    path = HERE / 'brimo_googleplaystore_review.csv'
    raw = pd.read_csv(path, usecols=['content', 'score'], low_memory=False)
    raw = raw.dropna(subset=['content'])
    raw['content'] = raw['content'].astype(str).str.strip()
    raw = raw[raw['content'].str.len() >= 8].drop_duplicates(subset=['content'])
    groups=[]
    for ratings in ([1, 2], [4, 5]):
        group=raw[raw['score'].isin(ratings)]
        if not group.empty:
            groups.append(group.sample(n=min(3,len(group)),random_state=42))
    if not groups:
        raise ValueError('Tidak ditemukan ulasan yang dapat digunakan sebagai sampel.')
    contoh=pd.concat(groups,ignore_index=True).rename(columns={'content':'ulasan'})[['ulasan']]
    return contoh.to_csv(index=False).encode('utf-8-sig')

st.download_button('📥 Download contoh data uji CSV', contoh_csv(), file_name='contoh_ulasan_baru.csv', mime='text/csv')
t1,t2=st.tabs(['Prediksi satu ulasan','Prediksi banyak CSV'])
with t1:
    st.caption('Ketik ulasan, kemudian tekan Enter di keyboard atau klik tombol Enter / Prediksi.')
    with st.form('form_prediksi_satu_ulasan'):
        u = st.text_input(
            'Masukkan ulasan aplikasi BRImo',
            placeholder='Contoh: Transfer sekarang lebih mudah dan cepat.',
        )
        submitted = st.form_submit_button('↵ Enter / Prediksi', type='primary')
    if submitted:
        if not u.strip():
            st.warning('Masukkan ulasan terlebih dahulu.')
        else:
            pred = info['kelas'][int(model.predict([u.strip()])[0])]
            conf = model.predict_proba([u.strip()])[0]
            if pred == 'Positif':
                st.success(f'Prediksi: **{pred}**')
            else:
                st.error(f'Prediksi: **{pred}**')
            st.bar_chart(pd.DataFrame({'probabilitas': conf}, index=info['kelas']))
with t2:
    f=st.file_uploader('Unggah CSV dengan kolom ulasan atau content',type='csv')
    if f:
        try:
            table = pd.read_csv(f)
            table.columns = table.columns.astype(str).str.strip().str.lstrip('\ufeff')

            # Dataset BRImo asli memakai 'content'; CSV contoh memakai 'ulasan'.
            if 'ulasan' not in table.columns and 'content' in table.columns:
                table = table.rename(columns={'content': 'ulasan'})

            if 'ulasan' not in table.columns:
                st.error('CSV harus mempunyai kolom ulasan atau content.')
            elif table.empty:
                st.warning('CSV tidak berisi data.')
            else:
                result = table.copy()
                texts = result['ulasan'].fillna('').astype(str).str.strip()
                # Baris kosong tidak perlu dikirim ke model.
                result['prediksi_sentimen'] = 'Tidak ada ulasan'
                valid = texts.ne('')
                if valid.any():
                    with st.spinner(f'Memprediksi {valid.sum():,} ulasan menggunakan DNN...'):
                        predictions = model.predict(texts[valid].tolist())
                    result.loc[valid, 'prediksi_sentimen'] = [info['kelas'][int(x)] for x in predictions]
                st.success(f'Selesai memproses {int(valid.sum()):,} ulasan.')
                # Tampilkan hanya kolom relevan agar balasan BRI/metadata tidak membingungkan.
                preview = result[['ulasan', 'prediksi_sentimen']].copy()
                preview.insert(0, 'No.', range(1, len(preview) + 1))
                st.dataframe(preview.head(100), width="stretch", hide_index=True)
                st.caption('Pratinjau 100 ulasan pertama; hasil unduhan mencakup semua baris.')
                st.download_button(
                    'Download hasil prediksi',
                    result.to_csv(index=False).encode('utf-8-sig'),
                    file_name='hasil_prediksi_dnn.csv',
                    mime='text/csv',
                )
        except Exception as exc: st.error(f'CSV gagal dibaca: {exc}')
st.caption('Catatan: label pelatihan diperoleh dari rating Google Play, bukan anotasi sentimen manual.')
