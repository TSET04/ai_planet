import json
import numpy as np
import soundfile as sf
import librosa
from vosk import Model, KaldiRecognizer
import easyocr
from PIL import Image
from logger import setup_logger

logger = setup_logger()

# ---------------- Load Vosk Model ----------------
vosk_model = Model("models/vosk/vosk-model-small-en-us-0.15")
logger.info("Vosk model loaded")

# ---------------- OCR Reader Cache ----------------
reader = None


# ---------------- OCR ----------------
def ocr_extract(img):
    global reader
    try:
        if reader is None:
            reader = easyocr.Reader(['en'], gpu=False)

        image = Image.open(img)
        image_np = np.array(image)

        result = reader.readtext(image_np)

        text = ' '.join([d[1] for d in result])
        confidence = (
            sum(d[2] for d in result) / len(result)
            if result else 0.0
        )

        return text, confidence

    except Exception as e:
        logger.exception("OCR failed")
        return "", 0.0


# ---------------- AUDIO → TEXT (FIXED) ----------------
def audio_to_text(audio_path):
    logger.info("Starting audio transcription")

    # 1️⃣ Load audio safely
    audio, sr = sf.read(audio_path, always_2d=False)

    # 2️⃣ Convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
        logger.info("Converted stereo to mono")

    # 3️⃣ Resample to 16kHz (CRITICAL)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
        logger.info("Resampled audio to 16kHz")

    # 4️⃣ Normalize amplitude
    max_amp = np.max(np.abs(audio))
    if max_amp < 0.01:
        logger.warning("Audio amplitude too low")
        return "", 0.0

    audio = audio / max_amp

    # 5️⃣ Convert float32 → int16 PCM (MANDATORY FOR VOSK)
    audio_int16 = (audio * 32767).astype(np.int16)

    logger.info(
        f"Audio stats | sr={sr}, duration={len(audio)/sr:.2f}s, max_amp={max_amp:.3f}"
    )

    # 6️⃣ Feed audio in chunks (recommended by Vosk)
    rec = KaldiRecognizer(vosk_model, sr)
    rec.SetWords(True)

    chunk_size = 4000
    for i in range(0, len(audio_int16), chunk_size):
        rec.AcceptWaveform(audio_int16[i:i+chunk_size].tobytes())

    result = json.loads(rec.FinalResult())

    text = normalize_math(result.get("text", "").strip())
    confidence = estimate_confidence(result)

    logger.info("ASR completed with confidence %.2f", confidence)
    return text, confidence


# ---------------- Text Normalization ----------------
def normalize_math(text):
    rules = {
        "square root of": "sqrt",
        "raised to the power": "^",
        "to the power of": "^",
        "divided by": "/",
        "times": "*",
        "into": "*",
        "minus": "-",
        "plus": "+"
    }

    text = text.lower()
    for k, v in rules.items():
        text = text.replace(k, v)

    return text


# ---------------- Confidence Estimation ----------------
def estimate_confidence(result):
    words = result.get("result", [])
    if not words:
        logger.warning("No words detected in ASR output")
        return 0.0

    return round(
        sum(w.get("conf", 0) for w in words) / len(words),
        2
    )
