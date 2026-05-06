# Klasifikasi Hewan Endemik Indonesia Menggunakan Deep Learning

Proyek ini merupakan implementasi model Deep Learning (Convolutional Neural Network) untuk mengklasifikasi 5 jenis hewan endemik Indonesia yang terancam punah. Sistem ini dibangun menggunakan framework **PyTorch** untuk pemrosesan gambar dan **FastAPI** sebagai antarmuka API, serta dikemas menggunakan **Docker** untuk kemudahan deployment.

Live Demo & Deployment
Model ini telah dideploy dan dapat diakses secara publik melalui Hugging Face Spaces:
https://tennocahyo-ai-image-klasifikasi-5-hewan-endemik.hf.space/docs

## Dokumentasi Penggunaan API (Endpoint)

API ini menyediakan endpoint untuk menerima gambar dan mengembalikan hasil klasifikasi AI dalam bentuk JSON. Model memiliki batas keyakinan (*confidence threshold*) sebesar 80%. Jika keyakinan model di bawah 80%, sistem akan mengklasifikasikan gambar sebagai "Tidak Dikenali".

### **1. Endpoint Prediksi**
* **URL:** `/predict`
* **Method:** `POST`
* **Content-Type:** `multipart/form-data`

### **2. Format Request (Input)**
Kirimkan file gambar menggunakan key `file`.
* **Key:** `file`
* **Value:** File gambar Anda (format `.jpg`, `.jpeg`, atau `.png`)

### **3. Format Response (Output JSON)**

```json
{
  "class_index": 4,
  "class_name": "Orangutan",
  "confidence": 92.5,
  "message": "Berhasil mendeteksi hewan!",
  "detail_probabilities": {
    "Bekantan": 1.2,
    "Beruang Madu": 0.5,
    "Burung Rangkong": 0.3,
    "Gajah": 5.5,
    "Orangutan": 92.5
  },
  "inference_ms": 45
}

{
  "class_index": -1,
  "class_name": "Bukan Hewan Tersebut (Tidak Dikenali)",
  "confidence": 65.2,
  "message": "Gambar kemungkinan besar bukan kelima hewan endemik tersebut.",
  "detail_probabilities": {
    "Bekantan": 15.0,
    "Beruang Madu": 10.5,
    "Burung Rangkong": 5.3,
    "Gajah": 4.0,
    "Orangutan": 65.2
  },
  "inference_ms": 42
}
