import streamlit as st
import numpy as np
from PIL import Image

def center_title(title):
    st.markdown(f"<h1 style='text-align:center;'>{title}</h1>", unsafe_allow_html=True)

# Fungsi untuk menampilkan halaman home
def home_page():
    st.title("🏠Home Page")
    st.markdown("Selamat datang di website kami!")

    # Load the pre-trained model
    model_path = 'best_model.h5'
    model = load_model(model_path)

    # Load sample submission data
    WORK_DIR = '../cassavaleafdiseaseclassification'

    # Set the target size
    TARGET_SIZE = 224

    # Mapping function to convert predicted class to disease label
    def map_class_to_disease(predicted_class):
        class_to_disease = {
            0: "Cassava Bacterial Blight (CBB)",
            1: "Cassava Brown Streak Virus Disease (CBSD)",
            2: "Cassava Green Mottle (CGM)",
            3: "Penyakit Mozaik Singkong (CMD)",
            4: "Daun Sehat (Healthy)"
        }
        return class_to_disease.get(predicted_class, "Kelas tidak dikenali")

    def preprocess_image(image):
        # Preprocess the image for prediction
        img = image.resize((TARGET_SIZE, TARGET_SIZE))
        img_array = np.expand_dims(img, axis=0)
        img_array = img_array / 255.0
        return img_array

    st.title("Prediksi Penyakit Daun Singkong")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Pilih gambar daun singkong...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the image
        img = Image.open(uploaded_file)
        img_array = preprocess_image(img)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        # Convert predicted class to disease label
        predicted_disease = map_class_to_disease(predicted_class)

        # Display the result
        st.image(img, caption="Gambar yang diunggah", use_column_width=True)
        
        # Hasil Prediksi dalam format kartu yang menarik
        st.markdown(
            f"""
            <div style="padding: 10px; border-radius: 10px; background-color: #f0f2f6; border: 1px solid #d1d5db; margin: 10px 0;">
                <h2 style="text-align: center;">Hasil Prediksi</h2>
                <p style="text-align: center; font-size: 18px;">Kategori: <strong>{predicted_disease}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Menampilkan array piksel dari gambar yang diunggah
        image_data = plt.imread(uploaded_file)
        image_data = image_data / 255.0
        red = image_data[:, :, 0]
        green = image_data[:, :, 1]
        blue = image_data[:, :, 2]

        def to_vector(matrix):
            return matrix.flatten()

        red_vector = to_vector(red)
        green_vector = to_vector(green)
        blue_vector = to_vector(blue)

        height, width, _ = image_data.shape
        vector_3d = np.stack((red_vector, green_vector, blue_vector), axis=-1).reshape(height, width, 3)

        # Visualisasi perubahan gambar menjadi array
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        axs[0].imshow(image_data)
        axs[0].set_title("Gambar Asli")

        axs[1].imshow(red, cmap='Reds')
        axs[1].set_title("Red Channel")

        axs[2].imshow(green, cmap='Greens')
        axs[2].set_title("Green Channel")

        axs[3].imshow(blue, cmap='Blues')
        axs[3].set_title("Blue Channel")

        st.pyplot(fig)

        # Display pixel arrays side by side
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write("Array Piksel(Red)")
            st.write(red)

        with col2:
            st.write("Array Piksel(Green)")
            st.write(green)

        with col3:
            st.write("Array Piksel(Blue)")
            st.write(blue)

        with col4:
            st.write("Vektor Gabungan")
            st.write(vector_3d)

# Fungsi untuk menampilkan halaman penyakit daun singkong
def projects_page():
    st.title("☘️Macam-macam Penyakit Daun Singkong")
    st.markdown("")
    st.markdown("**1. Cassava Bacterial Blight (CBB)**")
    st.image('cbb.jpg', caption='CBB')
    st.markdown("*Cassava bacterial blight (CBB) merupakan penyakit yang disebabkan oleh bakteri. Gejala yang terlihat jelas pada daun singkong antara lain hawar, layu, nekrosis dan mati pucuk. Bintik-bintik coklat nekrotik sudut menyebar terbatas pada bagian bawah daun. Setelah membesar dan menyatu, daunnya mati.*")  
    st.markdown("")
    st.markdown("**2. Cassava Brown Streak Virus Disease (CBSD)**")
    st.image('cbsd.jpg', caption='CBSD')
    st.markdown("*Penyakit cassava brown streak virus disease (CBSD) adalah penyakit mematikan pada singkong yang telah menyerang di Afrika Timur sejak tahun 1936. Penyakit ini ditandai dengan nekrosis dan klorosis hebat. Daun singkong yang terkena penyakit menunjukkan penampilan kekuningan dan berbintik-bintik.*")
    st.markdown("")
    st.markdown("**3. Cassava Green Mottle (CGM)**")
    st.image('cgm.jpg', caption='CGM')
    st.markdown("*Cassava green mottle (CGM) adalah penyakit tanaman yang disebabkan oleh tungau yang menyerang daun muda singkong. Tungau memasukkan bagian mulutnya yang menusuk ke dalam sel daun dan menghancurkan isi sel. Menyedot klorofil dari sel-sel yang menyebabkan daun berbintik-bintik dan mati.*")
    st.markdown("")
    st.markdown("**4. Penyakit Mozaik Singkong (CMD)**")
    st.image('cmd.jpg', caption='CMD')
    st.markdown("*Penyakit mozaik singkong (CMD), merupakan penyakit yang disebabkan oleh virus. Gejalanya ternyata terlihat pada daun singkong. Klorofil daun adalah sebaran mozaik termasuk penampilan daun singkong yang terdistorsi dan pertumbuhan singkong itu sendiri terhambat.*")
    st.markdown("")
    st.markdown("**5. Daun Sehat (Healthy)**")
    st.image('healthy.jpg', caption='Healthy')
    st.markdown("*Daun Sehat (Healthy) merupakan klasifikasi yang menunjukkan daun singkong yang sehat dan tidak terserang penyakit. Daun ini aman untuk dikonsumsi dan bisa menjadi indikator sehatnya tanaman singkong.*")

# Fungsi untuk menampilkan halaman about
def about_page():
    st.title("🔍About Page")
    st.markdown("**Sistem Klasifikasi Penyakit Daun Singkong**")  
    st.markdown("Sistem ini dikembangkan untuk mengklasifikasikan penyakit pada daun singkong berdasarkan tekstur daun. Menggunakan Convolutional Neural Network (CNN), sistem dapat memprediksi jenis penyakit dengan tingkat akurasi yang tinggi. Sistem ini dibuat guna memenuhi tugas akhir S1 Pendidikan Teknik Informatika dan Komputer, Universitas Negeri Semarang.")
    st.markdown("***")
    st.markdown("*Dikembangkan oleh Achmad Raja Qodli Zaka - NIM 5302419053*")

# Fungsi utama untuk menangani navigasi dan konten halaman
def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Pilih Halaman", ["Home", "Penyakit Daun Singkong", "About"])

    if selected_page == "Home":
        home_page()
    elif selected_page == "Penyakit Daun Singkong":
        projects_page()
    elif selected_page == "About":
        about_page()

if __name__ == "__main__":
    main()
