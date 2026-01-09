import streamlit as st
from multimodal import ocr_extract, audio_to_text
from agents import parser_agent, solver_agent, explainer_agent, verifier_agent
from rag import RAG
from hitl import needs_hitl
from memory import save_memory
from logger import setup_logger

logger = setup_logger()

st.title("Math Agent")

mode = st.selectbox("Input", ["Text", "Image", "Audio"])
text = ""

logger.info("User selected input mode: %s", mode)

if mode == "Text":
    text = st.text_area("Enter problem")

elif mode == "Image":
    img = st.file_uploader("Upload image")
    if img:
        text, conf = ocr_extract(img)
        st.text_area("OCR Text", text)
        st.write("Confidence:", conf)

elif mode == "Audio":
    audio = st.file_uploader("Upload audio")
    if audio:
        with open("data/audio.wav", "wb") as f:
            f.write(audio.read())
        text, conf = audio_to_text("data/audio.wav")
        st.text_area("Transcript", text)
        st.write("Confidence:", conf)

if st.button("Solve") and text:
    logger.info("Solve button clicked")
    parsed = parser_agent(text)

    if needs_hitl(0.9, parsed):
        st.warning("Needs clarification")
        logger.warning("Stopping pipeline due to HITL")
        st.stop()

    rag = RAG()
    context = rag.retrieve(parsed["problem_text"])
    solution = solver_agent(parsed["problem_text"], context)
    is_verified = verifier_agent(text, solution)
    if is_verified:
        explanation = explainer_agent(solution)
    else:
        explanation = "LLM Failed to Answer Correctly"

    st.subheader("Answer")
    st.write(solution)

    st.subheader("Explanation")
    st.write(explanation)

    save_memory({"input": text, "solution": solution, "result": is_verified})
    logger.info("Pipeline completed successfully")
