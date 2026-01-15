import streamlit as st
import tempfile, uuid
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

.chat-box { 
    border: 1px solid #dcdcdc; 
    padding: 12px; 
    margin-bottom: 6px; 
    color: #111; 
    font-size: 0.95rem; 
}

.user-box { background-color: #fcffd2; border-radius: 4px; }
.assistant-box { background-color: #ffffff; border-left: 4px solid #10b981; }
.clarification-box { background-color: #fff7e6; border-left: 4px solid #f59e0b; }
.warning-box { background-color: #fdecea; border-left: 4px solid #ef4444; }

textarea + div { display: none; }
.stForm { padding: 0px; } 
.stFileUploaderFileName { color: #131313; }
[data-testid="stWidgetLabel"] { color: #131313; }

.stExpander { background-color: #1a1c24; border-radius: 6px; }
[data-testid="stExpanderDetails"] {
    color: white;
    border: 2px solid black;
    border-radius: 0px 0px 6px 6px;
}    
[data-testid="stCaptionContainer"] { color: black; }
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
defaults = {
    "messages": [],
    "processing": False,
    "last_input": "",
    "show_image_upload": False,
    "show_audio_upload": False,
    "image_confidence": 1.0,
    "audio_confidence": 1.0,
    "clarification_history": [],
    "feedback": {}
}

for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---------------- Utilities ----------------
def safe_parse(text):
    try:
        out = parser_agent(text)
        if not isinstance(out, dict):
            return {"needs_clarification": True, "problem_text": text}
        return out
    except Exception as e:
        logger.warning(f"Parser failed: {e}")
        return {"needs_clarification": True, "problem_text": text}


def save_feedback(msg_id, feedback, message):
    save_memory({
        "type": "feedback",
        "message_id": msg_id,
        "feedback": feedback,
        "confidence": message.get("confidence"),
        "verified": message.get("verified"),
    })


# ---------------- Title ----------------
st.markdown("<h2 class='app-title'>Math Agent AI</h2>", unsafe_allow_html=True)

# ---------------- Chat Container ----------------
chat_container = st.container()
with chat_container:
    for m in st.session_state.messages:
        css = {
            "user": "user-box",
            "assistant": "assistant-box",
            "clarification": "clarification-box",
            "warning": "warning-box"
        }.get(m["role"], "assistant-box")

        # Message bubble (rendered ONCE)
        st.markdown(
            f"<div class='chat-box {css}'>{m['content']}</div>",
            unsafe_allow_html=True
        )

        # ----- Assistant metadata -----
        if m["role"] == "assistant":
            msg_id = m.get("id")
            feedback = st.session_state.feedback.get(msg_id)

            # -------- Confidence --------
            confidence = m.get("confidence")
            verified = m.get("verified")

            if confidence is not None:
                if confidence >= 50 and verified:
                    st.caption(f"Confidence: **{confidence}%**")
                else:
                    st.caption(f"⚠️ Low confidence · Confidence: **{confidence}%**")

            # -------- RAG Docs --------
            docs = m.get("rag_docs", [])
            if docs:
                with st.expander("RAG Docs"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.markdown(d)
                        st.markdown("---")

            # -------- Agent Trace --------
            trace = m.get("agent_trace")
            if trace:
                with st.expander("Agent Trace"):
                    st.json(trace)

            # -------- Feedback Buttons --------
            col1, col2, col3 = st.columns([1, 1, 6])

            if feedback is None:
                with col1:
                    if st.button("👍 Like", key=f"like_{msg_id}"):
                        st.session_state.feedback[msg_id] = "like"
                        save_feedback(msg_id, "like", m)
                        st.rerun()

                with col2:
                    if st.button("👎 Dislike", key=f"dislike_{msg_id}"):
                        st.session_state.feedback[msg_id] = "dislike"
                        save_feedback(msg_id, "dislike", m)
                        st.rerun()
            else:
                st.caption("Your feedback has been saved")
                                

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
            st.stop()

        st.session_state.audio_confidence = conf
        st.session_state.last_input = text
        st.session_state.show_audio_upload = False
        st.rerun()

# ---------------- Thinking Indicator ----------------
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

    user_input = user_input.strip()
    if not user_input:
        st.warning("Please enter a math problem before sending.")
        st.stop()

    st.session_state.processing = True
    st.session_state.last_input = ""

    thinking_placeholder.markdown(
        "<div class='chat-box assistant-box'><b>Thinking...</b></div>",
        unsafe_allow_html=True
    )

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    parsed = safe_parse(user_input)

    # ---------- Clarification ----------
    if parsed.get("needs_clarification", False):
        clarification = clarification_agent(parsed, st.session_state.clarification_history)

        st.session_state.clarification_history.append({
            "question": user_input,
            "answer": clarification.get("reason_for_clarification", "")
        })

        st.session_state.messages.append({
            "role": "clarification",
            "content": clarification.get("reason_for_clarification", "")
        })

        st.session_state.processing = False
        thinking_placeholder.empty()
        st.rerun()

    # ---------- HITL ----------
    min_conf = min(
        st.session_state.image_confidence,
        st.session_state.audio_confidence
    )

    if min_conf < 0.5 or needs_hitl(0.5, parsed):
        st.session_state.messages.append({
            "role": "warning",
            "content": f"Input unclear or low confidence ({min_conf:.2f}). Please rephrase."
        })
        st.session_state.processing = False
        thinking_placeholder.empty()
        st.rerun()

    # ---------- Solve ----------
    rag = RAG()
    rag_docs = rag.retrieve(parsed["problem_text"])

    solution = solver_agent(parsed["problem_text"], rag_docs)
    explanation = explainer_agent(solution)

    verification = verifier_agent(parsed["problem_text"], solution)
    print(verification)

    # ----- Confidence calibration (ONLY here) -----
    if verification["is_correct"]:
        verification["confidence"] = max(verification["confidence"], 30)

    verification["confidence"] = min(verification["confidence"], 100)

    agent_trace = {
        "parser_output": parsed,
        "rag_chunks_used": len(rag_docs),
        "raw_solution": solution,
        "verifier_result": verification
    }


    msg_id = str(uuid.uuid4())
    st.session_state.messages.append({
        "id": msg_id,  
        "role": "assistant",
        "content": explanation,
        "rag_docs": rag_docs,
        "confidence": verification["confidence"],
        "verified": verification["is_correct"],
        "agent_trace": agent_trace
    })

    save_memory({
        "input": user_input,
        "solution": solution,
        "confidence": verification["confidence"]
    })

    # ---------- Cleanup ----------
    st.session_state.processing = False
    thinking_placeholder.empty()
    st.session_state.image_confidence = 1.0
    st.session_state.audio_confidence = 1.0
    st.rerun()