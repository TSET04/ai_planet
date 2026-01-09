import cv2, json
import numpy as np
import soundfile as sf
from vosk import Model, KaldiRecognizer
import pytesseract
from logger import setup_logger

logger = setup_logger()

vosk_model = Model("models/vosk/vosk-model-small-en-us-0.15")
logger.info("Vosk model loaded")

def ocr_extract(uploaded_file):
    logger.info("Starting OCR extraction")
    bytes_data = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(bytes_data, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    text = " ".join(data["text"]).strip()

    confs = [int(c) for c in data["conf"] if c.isdigit() and int(c) > 0]
    confidence = sum(confs) / max(len(confs), 1) / 100

    logger.info("OCR completed with confidence %.2f", confidence)
    return text, confidence


def audio_to_text(audio_path):
    logger.info("Starting audio transcription")
    audio, sr = sf.read(audio_path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
        logger.info("Converted stereo audio to mono")

    rec = KaldiRecognizer(vosk_model, sr)
    rec.SetWords(True)
    rec.AcceptWaveform(audio.tobytes())

    result = json.loads(rec.FinalResult())
    text = normalize_math(result.get("text", ""))
    confidence = estimate_confidence(result)

    logger.info("ASR completed with confidence %.2f", confidence)
    return text, confidence


def normalize_math(text):
    rules = {
        "square root of": "sqrt",
        "raised to the power": "^",
        "divided by": "/",
        "into": "*",
        "minus": "-",
        "plus": "+"
    }
    for k, v in rules.items():
        text = text.replace(k, v)
    return text.lower()


def estimate_confidence(result):
    words = result.get("result", [])
    if not words:
        logger.warning("No words detected in ASR output")
        return 0.0
    return round(sum(w.get("conf", 0) for w in words) / len(words), 2)
