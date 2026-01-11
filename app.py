import streamlit as st
import tempfile, time
from multimodal import ocr_extract, audio_to_text
from agents import (
    parser_agent,
    solver_agent,
    explainer_agent,
    verifier_agent,
    clarification_agent
)
from rag import RAG
from hitl import needs_hitl
from memory import save_memory
from logger import setup_logger

logger = setup_logger()

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Math Agent",
    page_icon="🧮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f4f6f8; }
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 11rem; }
.app-title { text-align: center; font-weight: 700; color: #9de0ee !important; }
.chat-box { border: 1px solid #dcdcdc; padding: 12px; margin-bottom: 10px; color: #111; font-size: 0.95rem; }
.user-box { background-color: #fcffd2; border-radius: 4px; }
.assistant-box { background-color: #ffffff; border-left: 4px solid #10b981; }
.clarification-box { background-color: #fff7e6; border-left: 4px solid #f59e0b; }
.warning-box { background-color: #fdecea; border-left: 4px solid #ef4444; }
textarea + div { display: none; }
.stForm {padding:0px;} 
.stFileUploaderFileName {color: #131313;}
.stAlert {background-color: crimson;}
[data-testid="stWidgetLabel"] {color: #131313;}
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
defaults = {
    "messages": [],
    "processing": False,
    "awaiting_clarification": False,
    "original_problem": "",
    "last_input": "",
    "show_image_upload": False,
    "show_audio_upload": False,
    "image_confidence": 1.0,
    "audio_confidence": 1.0,
    "clarification_history": [],
}

for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---------------- Utilities ----------------
def safe_parse(text):
    """Always return a dict; fallback sets needs_clarification=True"""
    try:
        out = parser_agent(text)
        if not isinstance(out, dict):
            return {"needs_clarification": True, "problem_text": text}
        return out
    except Exception as e:
        logger.warning(f"Parser failed: {e}")
        return {"needs_clarification": True, "problem_text": text}

def typewriter(text, placeholder, delay=0.02, chunk_size=5):
    rendered = ""
    for i in range(0, len(text), chunk_size):
        rendered += text[i:i+chunk_size]
        placeholder.markdown(
            f"<div class='chat-box assistant-box'>{rendered}</div>",
            unsafe_allow_html=True
        )
        time.sleep(delay)

# ---------------- Title ----------------
st.markdown("<h2 class='app-title'>🧮 Math Agent AI</h2>", unsafe_allow_html=True)

# ---------------- Chat History ----------------
for m in st.session_state.messages:
    css = {
        "user": "user-box",
        "assistant": "assistant-box",
        "clarification": "clarification-box",
        "warning": "warning-box"
    }.get(m["role"], "assistant-box")

    st.markdown(
        f"<div class='chat-box {css}'>{m['content']}</div>",
        unsafe_allow_html=True
    )

# ---------------- Controls ----------------
c1, c2, _ = st.columns([1, 1, 6])
with c1:
    if st.button("🎤 Audio"):
        st.session_state.show_audio_upload = not st.session_state.show_audio_upload
        st.session_state.show_image_upload = False
with c2:
    if st.button("🖼️ Image"):
        st.session_state.show_image_upload = not st.session_state.show_image_upload
        st.session_state.show_audio_upload = False

# ---------------- Image Upload ----------------
if st.session_state.show_image_upload:
    img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if img:
        text, conf = ocr_extract(img)
        st.session_state.last_input = text
        st.session_state.image_confidence = conf
        st.session_state.show_image_upload = False
        st.rerun()

# ---------------- Audio Input ----------------
if st.session_state.show_audio_upload:
    audio = st.audio_input("Speak your problem")
    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio.getvalue())
            path = f.name

        text, conf = audio_to_text(path)

        if conf < 0.2 or not text.strip():
            st.warning("🎧 No clear speech detected. Please speak clearly and try again.")
            st.session_state.show_audio_upload = True
            st.stop()  # prevent further processing

        st.session_state.audio_confidence = conf
        st.session_state.last_input = text
        st.session_state.show_audio_upload = False
        st.rerun()

# ---------------- Thinking Indicator Placeholder ----------------
thinking_placeholder = st.empty()

# ---------------- Input Form ----------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Math input",
        value=st.session_state.last_input,
        placeholder="Type your math problem...",
        height=90,
        label_visibility="collapsed"
    )
    send = st.form_submit_button("Send")

# ---------------- Processing ----------------
if send and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.last_input = ""

    # Show thinking indicator immediately
    thinking_placeholder.markdown(
        "<div class='chat-box assistant-box'>🤔 <b>Thinking...</b></div>",
        unsafe_allow_html=True
    )

    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip()
    })

    parsed = safe_parse(user_input.strip())

    # ---------- Clarification ----------
    if parsed.get("needs_clarification", False):
        clarification = clarification_agent(parsed, st.session_state.clarification_history)

        # Add safe/normalized entry to clarification_history
        st.session_state.clarification_history.append({
            "question": user_input.strip(),
            "answer": clarification.get("reason_for_clarification", "")
        })

        st.session_state.messages.append({
            "role": "clarification",
            "content": clarification.get("reason_for_clarification", "")
        })
        st.session_state.processing = False
        st.rerun()


    # ---------- HITL ----------
    min_conf = min(
        st.session_state.image_confidence,
        st.session_state.audio_confidence
    )

    if min_conf < 0.85 or needs_hitl(0.9, parsed):
        st.session_state.messages.append({
            "role": "warning",
            "content": f"⚠️ Input unclear or low confidence ({min_conf:.2f}). Please rephrase."
        })
        st.session_state.processing = False
        thinking_placeholder.empty()
        st.rerun()

    # ---------- Solve ----------
    rag = RAG()
    ctx = rag.retrieve(parsed["problem_text"])
    solution = solver_agent(parsed["problem_text"], ctx)

    if verifier_agent(parsed["problem_text"], solution):
        explanation = explainer_agent(solution)

        anim = st.empty()
        typewriter(explanation, anim)

        st.session_state.messages.append({
            "role": "assistant",
            "content": explanation
        })

        save_memory({
            "input": user_input,
            "solution": solution
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ Unable to verify the solution."
        })

    st.session_state.processing = False
    thinking_placeholder.empty()