# Project Machine Learning: Volcano Eruption Prediction

Proyek ini berfokus pada pengembangan model klasifikasi untuk memprediksi probabilitas atau tipe aktivitas vulkanik berdasarkan fitur geofisika dan sensorik.

---

## Fitur Utama
* **Preprocessing Data:** Pembersihan data *missing values*, normalisasi, dan *feature engineering* pada dataset aktivitas gunung berapi.
* **Model Machine Learning:** Implementasi algoritma klasifikasi (seperti Random Forest, XGBoost, atau SVM) untuk prediksi akurat.
* **Evaluasi Performa:** Penggunaan metrik *Accuracy*, *Precision*, *Recall*, dan *F1-Score* serta *Confusion Matrix*.

---

## Arsitektur Proyek


1.  **Data Acquisition:** Mengambil data mentah dari sensor atau dataset historis.
2.  **Feature Extraction:** Menentukan variabel kunci seperti frekuensi seismik, emisi gas, dan deformasi tanah.
3.  **Training:** Melatih model pada dataset yang telah dipisah (*train-test split*).
4.  **Inference:** Melakukan prediksi terhadap data baru untuk peringatan dini.

---

## Teknologi yang Digunakan
* **Bahasa:** Python
* **Library:** 
    * `scikit-learn`: Untuk pemodelan machine learning.
    * `pandas` & `numpy`: Manipulasi dan analisis data.
    * `matplotlib` & `seaborn`: Visualisasi distribusi data dan hasil prediksi.

---

## Cara Penggunaan

### 1. Setup Lingkungan (Virtual Environment)
Disarankan untuk menggunakan virtual environment agar tidak mengganggu package global.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Menjalankan Aplikasi
Setelah dependensi terinstall, jalankan server FastAPI:
```bash
uvicorn main:app --reload
```
Aplikasi akan berjalan di `http://127.0.0.1:8000`.

---

## Docker Setup

Jika ingin menjalankan aplikasi menggunakan Docker:

1.  **Build Image:**
    ```bash
    docker build -t volcano-classifier .
    ```
2.  **Run Container:**
    ```bash
    docker run -p 8000:8000 volcano-classifier
    ```

---

## Deployment Guide

### 1. Deploy ke Google Cloud Run
Google Cloud Run sangat cocok untuk aplikasi containerized seperti ini.

1.  **Login ke Google Cloud:**
    ```bash
    gcloud auth login
    ```
2.  **Submit Build ke Cloud Build:**
    ```bash
    gcloud builds submit --tag gcr.io/[PROJECT_ID]/volcano-classifier
    ```
3.  **Deploy ke Cloud Run:**
    ```bash
    gcloud run deploy volcano-classifier --image gcr.io/[PROJECT_ID]/volcano-classifier --platform managed --allow-unauthenticated
    ```

### 2. Deploy ke AWS (App Runner)
AWS App Runner adalah cara termudah untuk menjalankan aplikasi web containerized di AWS.

1.  **Push ke Amazon ECR:**
    ```bash
    aws ecr get-login-password --region [REGION] | docker login --username AWS --password-stdin [ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com
    docker tag volcano-classifier:latest [ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com/volcano-classifier:latest
    docker push [ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com/volcano-classifier:latest
    ```
2.  **Buka AWS Console** -> App Runner -> Create Service.
3.  Pilih **Container registry** dan pilih image yang baru saja di-push.
4.  Atur Port ke `8000`.

### 3. Deploy ke Azure (App Service)
Gunakan Azure App Service untuk Containers.

1.  **Login ke Azure CLI:**
    ```bash
    az login
    ```
2.  **Push ke Azure Container Registry (ACR):**
    ```bash
    az acr login --name [ACR_NAME]
    docker tag volcano-classifier [ACR_NAME].azurecr.io/volcano-classifier:latest
    docker push [ACR_NAME].azurecr.io/volcano-classifier:latest
    ```
3.  **Create Web App:**
    ```bash
    az webapp create --resource-group [RG_NAME] --plan [PLAN_NAME] --name [APP_NAME] --deployment-container-image-name [ACR_NAME].azurecr.io/volcano-classifier:latest
    ```

### 4. Deploy ke Alibaba Cloud (SAE)
Serverless App Engine (SAE) adalah layanan serverless berbasis container di Alibaba Cloud.

1.  **Push ke Alibaba Cloud Container Registry (ACR):**
    ```bash
    docker login --username=[USERNAME] registry.[REGION].aliyuncs.com
    docker tag volcano-classifier registry.[REGION].aliyuncs.com/[NAMESPACE]/volcano-classifier:latest
    docker push registry.[REGION].aliyuncs.com/[NAMESPACE]/volcano-classifier:latest
    ```
2.  **Buka SAE Console** -> Create Application.
3.  Pilih **Image** sebagai deployment method dan pilih image dari ACR.
4.  Konfigurasi port HTTP ke `8000`.

### 5. LocalStack (Simulasi AWS Lokal)
LocalStack memungkinkan Anda untuk mensimulasikan layanan AWS di mesin lokal tanpa biaya.

1.  **Jalankan LocalStack via Docker:**
    ```bash
    docker run --rm -it -p 4566:4566 -p 4510-4559:4510-4559 localstack/localstack
    ```
2.  **Gunakan `awslocal` (AWS CLI Wrapper):**
    Install awslocal: `pip install awscli-local`
3.  **Simulasi ECR & Push Image:**
    ```bash
    awslocal ecr create-repository --repository-name volcano-classifier
    docker tag volcano-classifier localhost:4566/volcano-classifier:latest
    docker push localhost:4566/volcano-classifier:latest
    ```
4.  **Verifikasi Image di LocalStack:**
    ```bash
    awslocal ecr describe-images --repository-name volcano-classifier
    ```

### 6. Deploy ke VPS (Ubuntu/Debian)
Deployment manual ke VPS menggunakan Docker dan Nginx sebagai Reverse Proxy.

1.  **Install Docker di VPS:**
    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    ```
2.  **Clone & Build Image:**
    ```bash
    git clone https://github.com/Muhammad-Ikhwan-Fathulloh/Project-Machine-Learning-Volcano.git
    cd Project-Machine-Learning-Volcano
    docker build -t volcano-classifier .
    ```
3.  **Jalankan Container:**
    ```bash
    docker run -d --name volcano-api --restart always -p 8000:8000 volcano-classifier
    ```
4.  **Konfigurasi Nginx (Optional):**
    Buat file `/etc/nginx/sites-available/volcano-api`:
    ```nginx
    server {
        listen 80;
        server_name domain-anda.com;

        location / {
            proxy_pass http://localhost:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
    ```
    Aktifkan: `sudo ln -s /etc/nginx/sites-available/volcano-api /etc/nginx/sites-enabled/ && sudo systemctl restart nginx`

### 7. Deploy ke Sanberlify.app
Sanberlify (platform Sanbercode) mendukung deployment berbasis Git.

1.  Pastikan file `requirements.txt` dan `main.py` berada di root repository.
2.  Push code ke GitHub/GitLab.
3.  Buka dashboard [Sanberlify](https://sanberlify.com).
4.  Hubungkan repository Anda.
5.  Konfigurasi **Start Command** jika diminta:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
6.  Klik **Deploy**.

---

## Kontribusi
Kontribusi sangat terbuka. Silakan buka *issue* atau ajukan *pull request* untuk pengembangan fitur deteksi dini yang lebih lanjut.