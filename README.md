# Speech Denoising – Khử Nhiễu Tiếng Nói

## 🎵 Giới Thiệu

Đây là dự án **khử nhiễu tiếng nói (Speech Denoising)** sử dụng **xử lý tín hiệu số và Deep Learning**.  
Mục tiêu của dự án là loại bỏ **tiếng ồn nền** và giữ lại **giọng nói**, tương tự như cơ chế khử nhiễu của Discord.

Dự án được thực hiện trong khuôn khổ **đồ án học phần Xử lý tiếng nói**, triển khai và huấn luyện trực tiếp trên máy cá nhân.

---

## ✨ Chức Năng Chính

- 🎧 Khử nhiễu tiếng nói từ file audio
- 🖥️ Giao diện đồ họa (GUI) bằng `tkinter`
- 📊 Hiển thị waveform và spectrogram trước / sau khử nhiễu
- 📁 Xử lý nhiều file audio (batch processing)
- 🎓 Huấn luyện mô hình Deep Learning
- 📈 Đánh giá chất lượng bằng các metrics chuẩn (SNR, STOI, PESQ*)

> (*PESQ là tùy chọn, không bắt buộc cài trên Windows*)

---

## 🧠 Phương Pháp & Pipeline

Pipeline xử lý tiếng nói của hệ thống:

Audio (noisy)
↓
STFT
↓
Log-magnitude Spectrogram
↓
CNN Autoencoder (Speech Enhancement)
↓
Inverse STFT
↓
Audio (denoised)

yaml
Copy code

**Ý tưởng chính**:
- Mô hình **không học trực tiếp waveform**
- Chỉ học **biên độ phổ (magnitude)**
- Phase của tín hiệu nhiễu được giữ nguyên khi tái tạo âm thanh

---

## 📚 Dataset

### VoiceBank + DEMAND

Dataset được sử dụng rộng rãi trong nghiên cứu Speech Enhancement.

- Clean speech: **VoiceBank**
- Noise: **DEMAND**
- Sample rate: **16 kHz**

📥 Link dataset:  
https://datashare.ed.ac.uk/handle/10283/2791

### Cấu trúc thư mục sau khi giải nén:

speech_denoising/
├── data/
│ ├── clean_trainset_28spk_wav/
│ ├── noisy_trainset_28spk_wav/
│ ├── clean_testset_wav/
│ └── noisy_testset_wav/

yaml
Copy code

⚠️ **Dataset không được push lên GitHub** (đã ignore bằng `.gitignore`).

---

## 📂 Cấu Trúc Dự Án

speech_denoising/
├── app.py # GUI chính
├── run_app.py # Launcher GUI
├── config.yaml # Cấu hình training
├── train.py # Training script
├── inference.py # Khử nhiễu audio
├── evaluate.py # Đánh giá mô hình
├── demo.py # Demo nhanh
├── data/
│ └── dataset.py # Dataset loader
├── models/
│ ├── autoencoder.py # CNN Autoencoder
│ └── loss.py # Loss functions
├── utils/
│ ├── audio_utils.py
│ └── metrics.py
├── requirements.txt
├── README.md
└── .gitignore

yaml
Copy code

---

## 🖥️ Hướng Dẫn Chạy GUI

### 1️⃣ Cài dependencies

```bash
pip install -r requirements.txt
2️⃣ Chạy ứng dụng
bash
Copy code
python app.py
Hoặc:

bash
Copy code
python run_app.py
🎓 Training Mô Hình
Train với cấu hình mặc định
bash
Copy code
python train.py --config config.yaml
Resume training
bash
Copy code
python train.py \
  --config config.yaml \
  --resume checkpoints/model_epoch_20.pt
📊 Đánh Giá Mô Hình
bash
Copy code
python evaluate.py \
  --config config.yaml \
  --checkpoint checkpoints/best_model.pt
Metrics sử dụng:
Metric	Ý nghĩa
SNR	Signal-to-Noise Ratio
STOI	Độ dễ hiểu của tiếng nói
PESQ*	Chất lượng cảm nhận

⚙️ Cấu Hình (config.yaml – ví dụ)
yaml
Copy code
data:
  sample_rate: 16000
  segment_length: 32000   # 2 giây

stft:
  n_fft: 512
  hop_length: 128
  win_length: 512

training:
  batch_size: 16
  num_epochs: 30
  learning_rate: 0.0001
🚀 Ghi Chú Kỹ Thuật
Huấn luyện trên CPU laptop

Mô hình nhẹ (~1–3M parameters)

Thời gian train: ~2–4 giờ

Phù hợp cho đồ án học phần

👥 Làm Việc Nhóm
Branch chính: main

Mỗi thành viên làm việc trên feature/*

Merge vào main thông qua Pull Request

📖 Tài Liệu Tham Khảo
VoiceBank + DEMAND Dataset

U-Net for Speech Enhancement

Speech Enhancement using Autoencoder

Multi-Resolution STFT Loss

📜 License
MIT License

yaml
Copy code

---

## ✅ VIỆC BẠN CẦN LÀM NGAY

```bash
git add README.md
git commit -m "resolve README conflict"
git push