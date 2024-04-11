import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pickle

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Memanggil fungsi untuk menyematkan CSS
local_css("style.css")

URL = 'Data Cleaned (10).csv'
df = pd.read_csv(URL)

# Data referensi ID brand dan merk brand
brand_ref = {
    'cat': '1',
    'sony': '2',
    'blackberry': '3',
    'lg': '4',
    'asus': '5',
    'google': '6',
    'huawei': '7',
    'oneplus': '8',
    'motorola': '9',
    'nokia': '10',
    'apple': '11',
    'vivo': '12',
    'realme': '13',
    'oppo': '14',
    'xiaomi': '15',
    'samsung': '16'
}

# Membuat DataFrame dari data referensi
brand_df = pd.DataFrame(list(brand_ref.items()), columns=['Merk Brand', 'ID Brand'])

# Menghilangkan index default dari DataFrame
brand_df.set_index('Merk Brand', inplace=True)

# Fungsi untuk menampilkan Jumlah Unit Handphone Berdasarkan RAM
def plot_ram_distribution():
    st.title('Jumlah Unit Handphone Berdasarkan RAM')
    palette = sns.color_palette("Paired", len(df['RAM'].unique()))
    ax = sns.countplot(x='RAM', data=df, palette=palette)
    ax.set_xlabel('RAM')
    ax.set_ylabel('Count')
    plt.title('Jumlah Unit Handphone Berdasarkan RAM')
    st.pyplot()
    st.markdown("Berdasarkan visualisasi diatas didapatkan insight bahwa RAM 4 GB memiliki jumlah unit handphone paling banyak dibandingkan dengan kategori RAM lainnya")

# Fungsi untuk menampilkan Rata-rata Harga Handphone Berdasarkan RAM
def plot_avg_price_by_ram():
    st.title('Rata-rata Harga Handphone Berdasarkan RAM')
    avg_price_df = df.groupby('RAM')['Price'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='RAM', y='Price', data=avg_price_df, palette='coolwarm')
    plt.xlabel('RAM (GB)')
    plt.ylabel('Rata-rata Harga Handphone ($)')
    plt.title('Rata-rata Harga Handphone Berdasarkan RAM')
    plt.xticks(rotation=45)
    st.pyplot()
    st.markdown("Berdasarkan visualisasi diatas didapatkan insight bahwa harga handphone yang paling mahal dari RAM yaitu dari RAM 16 GB.")

# Fungsi untuk menampilkan Heatmap Korelasi Antar Fitur dengan Patokan Harga
def plot_correlation_heatmap():
    st.title('Heatmap Korelasi Antar Fitur dengan Patokan Harga')
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(corr_matrix[['Price']], annot=True, cmap='coolwarm', fmt=".2f")
    heatmap.set_title('Heatmap Korelasi Antar Fitur dengan Patokan Harga')
    st.pyplot()
    st.markdown("Berdasarkan visualisasi diatas didapatkan insight bahwa korelasi antar fitur paling mempengaruhi harga handphone yaitu fitur RAM dan Storage.")

# Menampilkan halaman utama dengan menu visualisasi
def main():
    st.sidebar.title('Menu')
    menu_options = ['Home', 'Visualisasi']
    choice = st.sidebar.selectbox('Pilih Menu', menu_options)

    if choice == 'Home':
        st.title('Welcome to Dashboard Prediksi Harga Handphone')
        st.subheader("Dataset Handphone yang telah dicleaning")
        st.write(df)

        model = pickle.load(open('dt_regressor.pkl', 'rb'))

        ScreenSize= st.number_input('Input nilai ScreenSize (inches)')
        BatteryCapacity = st.number_input('Input nilai Battery')
        RAM = st.number_input('Input nilai RAM')
        Storage = st.number_input('Input nilai storage')
        JumlahKamera = st.number_input('Input nilai jumlah kamera')
        cam1 = st.number_input('Input nilai cam1')
        cam2 = st.number_input('Input nilai cam2')
        cam3 = st.number_input('Input nilai cam3')
        cam4 = st.number_input('Input nilai cam4')
        idBrand = st.number_input('Input nilai id brand')

        predict = ''

        if st.button('prediksi Harga'):
            predict = model.predict(
                [[ScreenSize, BatteryCapacity, RAM, Storage, JumlahKamera, cam1, cam2, cam3, cam4, idBrand]]
            ) 
        st.write ('Prediksi Harga Handphone',predict)

    elif choice == 'Visualisasi':
        st.title('Menu Visualisasi')
        visualization_option = st.sidebar.selectbox(
            'Insight dari Visualisasi :',
            ('Jumlah Unit Handphone Berdasarkan RAM', 'Rata-rata Harga Handphone Berdasarkan RAM', 'Heatmap Korelasi Antar Fitur dengan Patokan Harga')
        )
        
        if visualization_option == 'Jumlah Unit Handphone Berdasarkan RAM':
            plot_ram_distribution()
        elif visualization_option == 'Rata-rata Harga Handphone Berdasarkan RAM':
            plot_avg_price_by_ram()
        elif visualization_option == 'Heatmap Korelasi Antar Fitur dengan Patokan Harga':
            plot_correlation_heatmap()
    
    st.sidebar.title('Tabel Referensi Merk Brand dan ID Brand')
    st.sidebar.table(brand_df)

if __name__ == "__main__":
    main()
