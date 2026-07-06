import requests
import json
import sys

# Konfigurasi URL API (Ganti dengan IP EC2 jika dideploy ke cloud, contoh: "http://your-ec2-ip:8080")
API_BASE_URL = "http://localhost:8080"

def print_separator(title):
    print("\n" + "=" * 60)
    print(f"  {title.upper()}  ")
    print("=" * 60)

def test_health():
    """Memeriksa kesehatan API dan status model"""
    print_separator("1. Cek Kesehatan API & Status Model (/health)")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=4))
            return True
        else:
            print(f"[ERROR] Server merespon dengan status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Tidak dapat terhubung ke API Server. {e}")
        return False

def test_single_prediction():
    """Menguji prediksi tunggal gunung berapi"""
    print_separator("2. Uji Prediksi Tunggal (/predict)")
    
    # Payload JSON sesuai dengan schema VolcanoInput
    payload = {
        "tinggi_meter": 2450.0,
        "lat": -7.25,
        "lon": 110.42
    }
    
    print(f"Mengirim JSON Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n[OK] Prediksi Berhasil!")
            print(f"   Shape Gunung Berapi : {result['prediction']}")
            print(f"   Tingkat Keyakinan   : {round(result['confidence'] * 100, 2)}%")
            print(f"   ID Log DynamoDB     : {result['input']['id']}")
            return result['input']['id']  # Mengembalikan log ID untuk pengetesan verifikasi nanti
        else:
            print(f"[ERROR] Gagal: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan: {e}")
        return None

def test_batch_prediction():
    """Menguji prediksi batch gunung berapi"""
    print_separator("3. Uji Prediksi Batch (/predict/batch)")
    
    # Payload JSON sesuai dengan schema VolcanoBatchInput
    payload = {
        "data": [
            {"tinggi_meter": 1500.0, "lat": -7.0, "lon": 110.0},
            {"tinggi_meter": 2801.0, "lat": 4.914, "lon": 96.329},
            {"tinggi_meter": 617.0, "lat": 5.820, "lon": 95.280}
        ]
    }
    
    print(f"Mengirim JSON Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n[OK] Batch Prediksi Berhasil!")
            print(f"   Total Berhasil Diproses: {result['total']}")
            print("\n   Hasil Prediksi:")
            for index, item in enumerate(result['results']):
                if item['status'] == 'success':
                    print(f"   [{index+1}] Tinggi: {item['input']['tinggi_meter']}m -> Prediksi: {item['prediction']} (Conf: {round(item['confidence']*100, 2)}%)")
                else:
                    print(f"   [{index+1}] Gagal memproses: {item.get('error')}")
        else:
            print(f"[ERROR] Gagal: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan: {e}")

def test_add_custom_sample():
    """Menambahkan data latih baru langsung ke DynamoDB dataset"""
    print_separator("4. Tambah Labeled Training Data (/add-training-data)")
    
    # Payload JSON sesuai dengan schema TrainingDataInput
    payload = {
        "tinggi_meter": 1810.0,
        "lat": 5.448,
        "lon": 95.658,
        "bentuk": "stratovulkan"
    }
    
    print(f"Mengirim JSON Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/add-training-data",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n[OK] Data Latih Baru Berhasil Disimpan ke DynamoDB!")
            print(f"   ID Data Baru: {result['id']}")
            print(f"   Pesan: {result['message']}")
        else:
            print(f"[ERROR] Gagal: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan: {e}")

def test_verify_prediction(log_id):
    """Mempromosikan log API biasa menjadi training sample di DynamoDB"""
    if not log_id:
        print("\n[INFO] Melewati pengujian verifikasi log karena tidak ada ID log yang tersedia.")
        return

    print_separator("5. Verifikasi & Promosikan Log ke Dataset (/verify-prediction)")
    
    # Payload JSON sesuai dengan schema VerifyLogInput
    payload = {
        "id": log_id,
        "bentuk": "stratovulkan"  # Mengoreksi atau memvalidasi label bentuk gunung
    }
    
    print(f"Mengirim JSON Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/verify-prediction",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n[OK] Log Inferensi Berhasil Diverifikasi Menjadi Dataset Latih!")
            print(f"   Pesan: {result['message']}")
        else:
            print(f"[ERROR] Gagal: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan: {e}")

def test_retraining():
    """Memicu pelatihan ulang model dan mengunggahnya ke S3 secara otomatis"""
    print_separator("6. Pemicuan Retraining Model & Upload ke S3 (/retrain)")
    print("Sedang melatih ulang Random Forest Classifier di server...")
    print("   (Proses ini mengambil dataset dasar, menggabungkan data kustom di DynamoDB, melatih model baru, mengunggah ke S3, dan melakukan hot-reload!)")
    
    try:
        response = requests.post(f"{API_BASE_URL}/retrain")
        
        if response.status_code == 200:
            result = response.json()
            print("\n[OK] RETRAINING BERHASIL DAN TER-HOT-RELOAD DI MEMORI API!")
            print(f"   Pesan                 : {result['message']}")
            print(f"   Skor Akurasi Training : {round(result['metrics']['training_score'] * 100, 2)}%")
            print(f"   Total Sampel Digunakan: {result['metrics']['total_samples']} (incl. {result['metrics']['custom_samples_used']} kustom dari DynamoDB)")
            print(f"   Daftar Kelas Gunung   : {', '.join(result['metrics']['classes'])}")
            
            # Print ringkasan klasifikasi report
            print("\n   Laporan Klasifikasi Model Terlatih Baru:")
            print(f"   {'Class Label':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
            print("   " + "-" * 60)
            
            for class_name, metrics in result['classification_report'].items():
                if class_name == 'accuracy':
                    continue
                print(f"   {class_name:<18} | {round(metrics['precision']*100, 1):<9}% | {round(metrics['recall']*100, 1):<9}% | {round(metrics['f1_score']*100, 1):<9}% | {metrics['support']:<8}")
        else:
            print(f"[ERROR] Retraining Gagal: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat retraining: {e}")

if __name__ == "__main__":
    print("============================================================")
    print("  MEMULAI PENGUJIAN API VOLCANO AI CLASSIFIER & AWS PIPELINE")
    print("============================================================")
    
    # 1. Cek kesehatan server API
    if not test_health():
        print("\n[ERROR] Pengujian dihentikan karena server API offline.")
        sys.exit(1)
        
    # 2. Uji Prediksi Tunggal dan ambil ID lognya
    log_id = test_single_prediction()
    
    # 3. Uji Prediksi Batch
    test_batch_prediction()
    
    # 4. Tambah sampel data latih kustom ke DynamoDB
    test_add_custom_sample()
    
    # 5. Verifikasi log tadi agar menjadi training data
    test_verify_prediction(log_id)
    
    # 6. Jalankan retraining model di server dan upload hasilnya ke S3
    test_retraining()
    
    print("\n" + "=" * 60)
    print("  SELURUH PENGUJIAN API AI DAN AWS FREE TIER Selesai!")
    print("=" * 60)
