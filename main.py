import torch
import torch.nn.functional as F
import librosa
import numpy as np
import gradio as gr
from pathlib import Path
from model import RawNet
import yaml

# --- 1. Load model config ---
config_path = "model_config_RawNet.yaml"  # YAML used for training
with open(config_path, 'r') as f:
    model_config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# --- 2. Initialize RawNet model ---
model_path = Path("Trained_model.pth")  # your trained model
if not model_path.exists():
    print(f"❌ Model file not found at '{model_path}'")
    exit()

model = RawNet(model_config['model'], device).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully")

# --- 3. Audio preprocessing ---
SAMPLE_RATE = 24000
MAX_LEN = SAMPLE_RATE * 4  # 4 seconds

def pad_audio(x, max_len=MAX_LEN):
    if len(x) >= max_len:
        return x[:max_len]
    repeats = int(np.ceil(max_len / len(x)))
    padded = np.tile(x, repeats)[:max_len]
    return padded

# --- 4. Prediction function ---
def predict_audio(audio_file):
    if not audio_file:
        return "Please upload an audio file."
    try:
        y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
        y_pad = pad_audio(y)
        x_tensor = torch.tensor(y_pad, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out_binary, out_multi = model(x_tensor)
            probs = F.softmax(out_binary, dim=1)
            return {"Fake": float(probs[0,0]), "Real": float(probs[0,1])}
    except Exception as e:
        return f"Error during prediction: {e}"

# --- 5. Launch Gradio ---
iface = gr.Interface(
    fn=predict_audio,
    inputs=[gr.Audio(type="filepath", label="Upload Audio File")],
    outputs=gr.Label(label="Prediction Probability"),
    title="Audio Deepfake Detection (RawNet)",
    description="Upload a 4s audio clip to classify as REAL or FAKE.",
    flagging_mode="never"
)

iface.launch()
