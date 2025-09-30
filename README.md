# Audio Deepfake Detection

This project is a **Deepfake Audio Detection** system that classifies audio clips as **REAL** or **FAKE** using deep learning models. It leverages **RawNet** and **CRNN-based models** trained on LibriSeVoc dataset for audio classification.



## 🔗 Project Structure

```
deepfake_audio/
│
├─ model.py # RawNet model definition
├─ model_config_RawNet.yaml # Model hyperparameters
├─ Trained_model.pth # Pretrained model (download from link given below)
├─ requirements.txt # Python dependencies
├─ .gitignore # Ignore venv, models, logs
└─ README.md
```

---

## ⚡ Prerequisites

- Python 3.10+  
- pip or conda  

---

## 🛠️ Setup Virtual Environment

### **Mac / Linux**

```
cd /path/to/AudioDetect
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### **Windows**

```
cd \path\to\AudioDetect
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```



## 📦 Download Trained Model

You can download the pre-trained model from the following link:

- [Trained_model.pth](https://drive.google.com/file/d/14WZSAwzkyldrQF2nvHd4PHB2UpJ6cNoK/view?usp=sharing)



##**Run the Project**

Run the Gradio interface to classify audio files:

```python main.py ```or ```python3 main.py```


##**💡 References**

## 💡 References

- [RawNet Paper](https://arxiv.org/abs/1810.11472)
- [LibriSeVoc Dataset](https://drive.google.com/file/d/1zHh7d6jxyzExample/view?usp=sharing)

