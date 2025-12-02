import streamlit as st
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import sounddevice as sd
import tempfile
import os
import wave
import json
import pyttsx3
import socket

# Extra imports for sign detection
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model as keras_load_model
from collections import deque, Counter

# -------------------------------------------------
# BASIC PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Voice & Sign Translation", layout="wide")

# -------------------------------------------------
# SIDEBAR MODULE SELECTION
# -------------------------------------------------
page = st.sidebar.radio(
    "Select Module",
    ["Voice Translation", "Sign Language Detection"]
)

# -------------------------------------------------
# COMMON / VOICE TRANSLATION PART (UNCHANGED LOGIC)
# -------------------------------------------------

# Offline imports
try:
    from vosk import Model, KaldiRecognizer
    from transformers import MarianMTModel, MarianTokenizer
except ImportError:
    Model = None
    KaldiRecognizer = None
    MarianMTModel = None
    MarianTokenizer = None

# ----------------- Languages -----------------
ONLINE_LANGS = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Punjabi": "pa",
    "Marathi": "mr"
}
SPEECH_LANGS = {lang: f"{code}-IN" for lang, code in ONLINE_LANGS.items()}

OFFLINE_LANGS = {
    "English": "en",
    "Hindi": "hi"
}

VOSK_MODELS = {
    "English": r"D:\SVVV\projects\real time voice translation\vosk-model-small-en-us-0.15",
    "Hindi": r"D:\SVVV\projects\real time voice translation\vosk-model-small-hi-0.22"
}

# Check offline models
for lang, path in VOSK_MODELS.items():
    if not os.path.exists(path):
        st.warning(f"⚠️ Vosk offline model for {lang} not found at: {path}")

# ----------------- Initialize pyttsx3 -----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ----------------- Audio Device Setup -----------------
devices = sd.query_devices()
input_devices = [d for d in devices if d["max_input_channels"] > 0]
default_input_index = 1
input_device_names = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d["max_input_channels"] > 0]
input_device_indices = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]

# ----------------- MarianMT Cache -----------------
marian_tokenizers = {}
marian_models = {}

def load_marian_model(src, tgt):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    if (src, tgt) not in marian_models:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        marian_tokenizers[(src, tgt)] = tokenizer
        marian_models[(src, tgt)] = model
    return marian_tokenizers[(src, tgt)], marian_models[(src, tgt)]

def offline_translate(text, src, tgt):
    if src == tgt or MarianMTModel is None:
        return text
    try:
        tokenizer, model = load_marian_model(src, tgt)
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Offline translation error: {e}")
        return text

# ----------------- Offline TTS -----------------
def offline_tts_pyttsx3(text):
    try:
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.close()
        engine.save_to_file(text, temp_wav.name)
        engine.runAndWait()
        return temp_wav.name
    except Exception as e:
        st.error(f"Offline TTS failed: {e}")
        return None

# ----------------- Internet Check -----------------
def is_internet_available(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

# ----------------- Audio Recording -----------------
def record_audio(duration_sec=5, fs=16000, device_id=None):
    try:
        recording = sd.rec(int(duration_sec*fs), samplerate=fs, channels=1, dtype='int16', device=device_id)
        sd.wait()
        return recording.reshape(-1)
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None

def save_wav(filename, recording, fs=16000):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())

# ----------------- Offline Speech Recognition -----------------
def offline_speech_to_text(lang, wav_file):
    model_path = VOSK_MODELS.get(lang)
    if not model_path or not os.path.exists(model_path):
        st.error(f"Vosk model for {lang} not found at {model_path}")
        return ""
    
    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    try:
        with wave.open(wav_file, "rb") as wf:
            text = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text += " " + result.get("text", "")
            final_result = json.loads(rec.FinalResult())
            text += " " + final_result.get("text", "")
        return text.strip()
    except Exception as e:
        st.error(f"Offline recognition failed: {e}")
        return ""

# =================================================
# PAGE 1: VOICE TRANSLATION (YOUR ORIGINAL LOGIC)
# =================================================
if page == "Voice Translation":

    st.title("🎙️ Real-Time Voice Translator")
    mode = st.radio("Select Mode:", ["Online (Internet Required)", "Offline (No Internet)"])

    # Session state
    if "input_lang" not in st.session_state:
        st.session_state.input_lang = "Hindi"
    if "output_lang" not in st.session_state:
        st.session_state.output_lang = "English"

    # Language selection
    if mode == "Online (Internet Required)":
        st.session_state.input_lang = st.selectbox(
            "Input Language", 
            list(ONLINE_LANGS.keys()), 
            index=list(ONLINE_LANGS.keys()).index(st.session_state.input_lang)
        )
        st.session_state.output_lang = st.selectbox(
            "Output Language", 
            list(ONLINE_LANGS.keys()), 
            index=list(ONLINE_LANGS.keys()).index(st.session_state.output_lang)
        )
    else:
        st.session_state.input_lang = st.selectbox(
            "Input Language", 
            list(OFFLINE_LANGS.keys()), 
            index=list(OFFLINE_LANGS.keys()).index(st.session_state.input_lang)
        )
        st.session_state.output_lang = st.selectbox(
            "Output Language", 
            list(OFFLINE_LANGS.keys()), 
            index=list(OFFLINE_LANGS.keys()).index(st.session_state.output_lang)
        )

    # Swap button
    if st.button("🔄 Swap Languages"):
        st.session_state.input_lang, st.session_state.output_lang = (
            st.session_state.output_lang,
            st.session_state.input_lang,
        )

    # Microphone selection
    st.markdown("### Select Input Microphone Device")
    selected_device_str = st.selectbox(
        "Input Device",
        input_device_names,
        index=(
            input_device_indices.index(default_input_index)
            if default_input_index in input_device_indices
            else 0
        ),
    )
    selected_device_index = int(selected_device_str.split(":")[0])

    # Record duration
    duration = st.slider("Record Duration (seconds)", 3, 10, 5)

    # Start Recording & Translate
    if st.button("Start Recording & Translate"):

        st.info("🎙️ Recording...")
        recording = record_audio(duration_sec=duration, device_id=selected_device_index)
        if recording is None:
            st.error("Recording failed.")
        else:
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_wav.close()
            save_wav(temp_wav.name, recording)

            # ---------- Online Mode ----------
            if mode == "Online (Internet Required)":
                recognizer = sr.Recognizer()
                with st.spinner("🌐 Recognizing & Translating..."):
                    try:
                        with sr.AudioFile(temp_wav.name) as source:
                            audio_data = recognizer.record(source)
                            recognized_text = recognizer.recognize_google(
                                audio_data, language=SPEECH_LANGS[st.session_state.input_lang]
                            )
                            st.info(f"📝 Recognized Text ({st.session_state.input_lang}): {recognized_text}")

                            translated_text = GoogleTranslator(
                                source=ONLINE_LANGS[st.session_state.input_lang],
                                target=ONLINE_LANGS[st.session_state.output_lang],
                            ).translate(recognized_text)
                            st.info(f"💬 Translated Text ({st.session_state.output_lang}): {translated_text}")

                            tts = gTTS(translated_text, lang=ONLINE_LANGS[st.session_state.output_lang])
                            mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                            tts.save(mp3_path)
                            audio_bytes = open(mp3_path, "rb").read()
                            st.audio(audio_bytes, format="audio/mp3")
                            os.remove(mp3_path)

                    except Exception as e:
                        st.error(f"Online recognition failed: {e}")

            # ---------- Offline Mode ----------
            else:
                recognized_text = offline_speech_to_text(
                    st.session_state.input_lang,
                    temp_wav.name  # pass the recorded WAV file path
                )

                if recognized_text:
                    st.info(f"📝 Recognized Text ({st.session_state.input_lang}): {recognized_text}")

                    src_code = OFFLINE_LANGS[st.session_state.input_lang]
                    tgt_code = OFFLINE_LANGS[st.session_state.output_lang]

                    translated_text = offline_translate(recognized_text, src_code, tgt_code)
                    st.info(f"💬 Translated Text ({st.session_state.output_lang}): {translated_text}")

                    # Check internet for Google TTS
                    if is_internet_available():
                        try:
                            tts = gTTS(translated_text, lang=OFFLINE_LANGS[st.session_state.output_lang])
                            mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                            tts.save(mp3_path)
                            audio_bytes = open(mp3_path, "rb").read()
                            st.audio(audio_bytes, format="audio/mp3")
                            os.remove(mp3_path)
                        except Exception as e:
                            st.error(f"Google TTS failed: {e}")
                    else:
                        st.warning("⚠️ No internet connection. Only text is displayed.")

            # Clean temp wav
            if os.path.exists(temp_wav.name):
                os.remove(temp_wav.name)

# =================================================
# PAGE 2: SIGN LANGUAGE DETECTION
# =================================================
if page == "Sign Language Detection":

    # ---------------------- CONFIG ----------------------
    ACTIONS = np.array(['hello', 'thankyou', 'iloveyou', 'yes', 'no'])
    MODEL_PATH = 'action_model.h5'      # make sure this file exists in same folder
    SEQUENCE_LENGTH = 30                # frames to collect before prediction
    CONF_THRESHOLD = 0.8                # confidence threshold
    SMOOTHING_WINDOW = 7                # number of recent predictions to smooth
    # ----------------------------------------------------

    st.title("🤟 Real-Time Sign Language Detection")
    st.write("Model detects: **hello, thankyou, iloveyou, yes, no**")

    @st.cache_resource
    def load_sign_model(path):
        return keras_load_model(path)

    sign_model = load_sign_model(MODEL_PATH)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    def mediapipe_detection(image, holistic_model):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic_model.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return image_bgr, results

    def draw_styled_landmarks(image, results):
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z]
                         for res in getattr(results, "pose_landmarks", []).landmark]
                         ).flatten() if results.pose_landmarks else np.zeros(33 * 3)
        face = np.array([[res.x, res.y, res.z]
                         for res in getattr(results, "face_landmarks", []).landmark]
                         ).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z]
                       for res in getattr(results, "left_hand_landmarks", []).landmark]
                       ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z]
                       for res in getattr(results, "right_hand_landmarks", []).landmark]
                       ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])

    # session state for sign module
    if "sign_sequence" not in st.session_state:
        st.session_state.sign_sequence = []
    if "sign_pred_buffer" not in st.session_state:
        st.session_state.sign_pred_buffer = deque(maxlen=SMOOTHING_WINDOW)
    if "sign_running" not in st.session_state:
        st.session_state.sign_running = False

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Controls")
        if not st.session_state.sign_running:
            if st.button("▶ Start Detection"):
                st.session_state.sign_running = True
        else:
            if st.button("⏹ Stop Detection"):
                st.session_state.sign_running = False

        st.markdown(f"**Sequence length:** {SEQUENCE_LENGTH}")
        st.markdown(f"**Confidence threshold:** {CONF_THRESHOLD}")
        st.markdown(f"**Smoothing window:** {SMOOTHING_WINDOW}")

    with col1:
        frame_placeholder = st.empty()

    if st.session_state.sign_running:
        cap = cv2.VideoCapture(0)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            while st.session_state.sign_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Could not read from webcam.")
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                keypoints = extract_keypoints(results)
                st.session_state.sign_sequence.append(keypoints)
                st.session_state.sign_sequence = st.session_state.sign_sequence[-SEQUENCE_LENGTH:]

                final_label = ""
                res = None

                if len(st.session_state.sign_sequence) == SEQUENCE_LENGTH:
                    res = sign_model.predict(
                        np.expand_dims(st.session_state.sign_sequence, axis=0),
                        verbose=0
                    )[0]
                    pred_idx = np.argmax(res)
                    pred_label = ACTIONS[pred_idx]
                    conf = res[pred_idx]

                    st.session_state.sign_pred_buffer.append(pred_label)
                    most_common, count = Counter(st.session_state.sign_pred_buffer).most_common(1)[0]

                    if conf > CONF_THRESHOLD and count > 3:
                        final_label = most_common

                    cv2.rectangle(image, (0, 0), (640, 50), (245, 117, 16), -1)
                    cv2.putText(
                        image,
                        final_label.upper(),
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    for i, act in enumerate(ACTIONS):
                        prob = float(res[i])
                        y1 = 60 + i * 40
                        y2 = y1 + 30
                        cv2.rectangle(image, (10, y1), (10 + int(prob * 200), y2), (0, 255, 0), -1)
                        cv2.putText(
                            image,
                            f"{act}: {prob:.2f}",
                            (220, y1 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                frame_placeholder.image(image, channels="BGR")

                if not st.session_state.sign_running:
                    break

        cap.release()
    else:
        st.info("Click **Start Detection** to begin sign language recognition using your webcam.")
