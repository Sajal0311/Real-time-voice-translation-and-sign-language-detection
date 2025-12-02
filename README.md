Real-Time Voice Translation & Sign Language Detection System

This project integrates real-time multilingual voice translation, offline speech recognition, and AI-based sign language detection into a single accessible communication platform. It combines Google Speech API, VOSK offline ASR, MarianMT translation, MediaPipe Holistic, and an LSTM deep-learning model, all wrapped in an easy-to-use Streamlit web application.

🚀 Features

🎤 Voice Translation

 * Online translation: Uses Google API for high-accuracy multilingual speech translation.

* Offline translation: Uses VOSK + MarianMT for translation without internet.

* Converts translated text into speech output using Google TTS / pyttsx3.

🤟 Sign Language Detection

* Uses MediaPipe Holistic to extract body & hand keypoints.

* LSTM deep-learning model recognizes 5 signs:

* Hello, Thank You, I Love You, Yes, No

* Real-time webcam-based sign recognition.

🌐 Web Interface

* Built with Streamlit

* Simple UI for recording audio, selecting languages, and performing gesture detection.

* Works on Windows/Linux/Mac with a webcam.

Project Structure

├── main.py                     # Streamlit main application

├── app.py                      # Standalone sign language detection script

├── action_model.h5             # Trained LSTM model for gesture recognition

├── vosk-model-small-en-us/     # Offline English ASR model

├── vosk-model-small-hi/        # Offline Hindi ASR model

├── requirements.txt            # Required Python packages

├── README.md                   # Documentation

└── dataset/                    # Keypoints dataset for LSTM (optional)

🔧 Installation

1. Clone the repository

* git clone https://github.com/yourusername/yourrepo.git
* cd yourrepo

2. Install required libraries

* pip install -r requirements.txt

3. Download VOSK models

* Place them in a folder:

* vosk-model-small-en-us-0.15

* vosk-model-small-hi-0.22

* Update the VOSK path in main.py if needed.

4. Run the Streamlit app

* streamlit run main.py

🧠 Tech Stack

Languages & Frameworks

 * Python 3.10

 * Streamlit

 * TensorFlow / Keras

 * MediaPipe

 * VOSK

 * MarianMT

 * Google Translator API

 * OpenCV

 * Models Used

 * LSTM model for gesture classification

 * VOSK small models for offline ASR

 * Transformer-based MarianMT for offline translation

🧪 How to Use

🎤 Voice Translation

1. Choose Online or Offline mode.

2. Select input & output languages.

3. Record your voice.

4. The app translates and speaks the output.

🤟 Sign Language Detection

1. Click Start Sign Detection.

2. Show the gesture clearly to the webcam.

3. The system will display the detected sign + confidence.

📸 Screenshots

<img width="569" height="729" alt="{9B477676-52BA-46CE-BBD3-21F25EE151F4}" src="https://github.com/user-attachments/assets/e1c5252d-3fea-4656-adaa-3d1d6831e55e" />

<img width="608" height="770" alt="{0762B70C-E9C3-43ED-8029-5116BDA15AEF}" src="https://github.com/user-attachments/assets/8899bf0c-432d-4edd-96f9-5129bdcf160f" />

<img width="454" height="509" alt="{E532A35D-ECE8-49CF-86B3-CFAEEACB6F68}" src="https://github.com/user-attachments/assets/cf26d18b-fa34-42f1-a0f2-e45995936283" />


