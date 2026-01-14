import json
import numpy as np
import soundfile as sf
import easyocr
import whisper
from PIL import Image
from logger import setup_logger

logger = setup_logger()

logger.info("Loading Whisper tiny model...")
whisper_model = whisper.load_model("tiny") 
logger.info("Whisper model loaded")

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
    """
    Transcribe audio using Whisper tiny model.

    Returns:
        text (str): transcribed text
        confidence (float): approximate confidence (0-1)
    """
    logger.info("Starting audio transcription with Whisper")

    try:
        # Whisper prefers 16kHz mono, read with soundfile
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # stereo -> mono

        # Save temporary file in WAV format if needed
        temp_wav = audio_path  # can reuse uploaded path

        # Transcribe
        result = whisper_model.transcribe(temp_wav, language="en", fp16=False)

        text = result.get("text", "").strip().lower()
        # Use logprob to estimate confidence if available
        segments = result.get("segments", [])
        if segments:
            avg_conf = sum(seg.get("avg_logprob", 0) for seg in segments) / len(segments)
            confidence = float(np.exp(avg_conf))  # convert logprob to 0-1 approx
        else:
            confidence = 0.9 if text else 0.0

        # Optional: normalize math terms
        text = normalize_math(text)

        logger.info("Whisper ASR completed with confidence %.2f", confidence)
        return text, confidence

    except Exception as e:
        logger.error("Whisper ASR failed: %s", str(e))
        return "", 0.0


# ---------------- Text Normalization ----------------
def normalize_math(text):
    rules = {
        "unde root": "√",
        "raised to the power": "^",
        "to the power of": "^",
        "divided by": "/",
        "times": "*",
        "into": "*",
        "minus": "-",
        "plus": "+",
        "x square": "x^2",
        "x cube": "x^3",
        "equals": "=",
        "equal": "=",
        "is equal to": "=",
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
