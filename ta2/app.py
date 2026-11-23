import uuid
import html
import requests
import streamlit as st

from essay_assistant import graph
from rag_memory import search_memory, save_memory, collection

st.set_page_config(
    page_title="EssayForge | AI Writing Studio",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Visual theme
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --navy:#14263f; --navy2:#203b56; --teal:#159b91; --ink:#192943; --muted:#687b96; }
html, body, [class*="css"] {font-family:'DM Sans',sans-serif}
.stApp {background:#f3f6f7;color:var(--ink)}
.block-container {max-width:1320px;padding-top:1.5rem;padding-bottom:3rem}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#14263f,#203b56)}
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stMarkdown strong {color:#f3f7fb}
[data-testid="stSidebar"] * {color:#f3f7fb}
/* Sidebar cards need contrast against the dark sidebar. */
[data-testid="stSidebar"] .step {background:rgba(255,255,255,.09) !important;border:1px solid rgba(255,255,255,.16) !important;color:#f3f7fb !important}
[data-testid="stSidebar"] .step b, [data-testid="stSidebar"] .step small {color:#f3f7fb !important}
[data-testid="stSidebar"] .memory-item {background:rgba(255,255,255,.09) !important;border:1px solid rgba(255,255,255,.16) !important;color:#f3f7fb !important}
[data-testid="stSidebar"] [data-testid="stCaption"] {color:#c1cfdb !important}
[data-testid="stSidebar"] .stCaption {color:#c1cfdb}
.hero {padding:2.2rem 2.35rem;border-radius:24px;background:linear-gradient(115deg,#14263f 0%,#21445c 62%,#176e70 100%);color:white;margin-bottom:1.5rem;box-shadow:0 18px 45px #14263f20;position:relative;overflow:hidden}
.hero .eyebrow {color:#8ce0d4}
.hero h1 {font-family:'Manrope',sans-serif;font-size:2.45rem;line-height:1.15;margin:.55rem 0}
.hero p {color:#dce8ef;font-size:1.03rem;max-width:760px;margin:.65rem 0 0}
.eyebrow {color:#70829d;font-size:.72rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase}
.panel {background:#fff;border:1px solid #e1e9ed;border-radius:19px;padding:1.25rem 1.4rem;margin-bottom:1rem;box-shadow:0 8px 28px #17334b0a}
.panel h3 {color:var(--ink);font-family:'Manrope',sans-serif;margin:.4rem 0 .7rem}
.muted {color:var(--muted);font-size:.92rem;line-height:1.65}
.step {border-radius:12px;padding:.8rem .9rem;background:#f4f7f8;border:1px solid #e3ebee;margin:.45rem 0;color:#243a50}
.pill {display:inline-block;border:1px solid #ffffff35;background:#ffffff16;border-radius:30px;padding:.45rem .8rem;color:#e8f4f5;font-size:.8rem;font-weight:600;margin-top:1rem}
.stButton>button[kind="primary"] {background:linear-gradient(90deg,#138f88,#20aa9b);border:0;border-radius:11px;font-weight:700;min-height:3rem;color:white}
.stButton>button {border-radius:10px}
/* Explicit input/output contrast: Streamlit may inherit a dark browser theme. */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background-color:#ffffff !important;
    color:#192943 !important;
    -webkit-text-fill-color:#192943 !important;
    caret-color:#138f88 !important;
    border:1px solid #cbd8e2 !important;
    border-radius:11px !important;
    opacity:1 !important;
}
div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stTextInput"] input::placeholder,
[data-baseweb="input"] input::placeholder {
    color:#718198 !important;
    -webkit-text-fill-color:#718198 !important;
    opacity:1 !important;
}
/* Keep labels and expander controls readable in both light and dark themes. */
div[data-testid="stTextArea"] label,
div[data-testid="stTextInput"] label {color:#34465f !important}
/* Streamlit expander headers can inherit the browser's dark theme. Force a
   light header with dark text, including nested spans and icons. */
div[data-testid="stExpander"] details > summary,
div[data-testid="stExpander"] details > summary:hover,
div[data-testid="stExpander"] details > summary:focus {
    background:#eaf1f5 !important;
    color:#192943 !important;
    border:1px solid #d5e0e8 !important;
    border-radius:10px !important;
}
div[data-testid="stExpander"] details > summary * {
    color:#192943 !important;
    fill:#192943 !important;
    -webkit-text-fill-color:#192943 !important;
}
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background:#ffffff !important;
    color:#192943 !important;
}
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
    color:inherit;
}
/* High-contrast status messages: Streamlit alerts must never use dark text on saturated fills. */
[data-testid="stAlert"], .stAlert {
    border-radius:12px !important;
    border:1px solid #cfe3e8 !important;
    background:#edf5f8 !important;
    color:#19334a !important;
}
[data-testid="stAlert"] *, .stAlert * {
    color:#19334a !important;
    -webkit-text-fill-color:#19334a !important;
    opacity:1 !important;
}
/* Success messages (memory retrieved, critique passed, revision completed). */
[data-testid="stAlert"]:has([data-testid="stIconSuccess"]),
.stAlert:has([data-testid="stIconSuccess"]) {
    background:#e7f5f1 !important;
    border-color:#b8e4d8 !important;
}
/* Keep the success icon legible while the message text remains dark teal. */
[data-testid="stAlert"] [data-testid="stIconSuccess"] *,
.stAlert [data-testid="stIconSuccess"] * {color:#087765 !important;fill:#087765 !important}
/* Info and warning variants also use readable light surfaces. */
[data-testid="stAlert"]:has([data-testid="stIconInfo"]),
.stAlert:has([data-testid="stIconInfo"]) {background:#edf3fb !important;border-color:#d4e1f1 !important}
[data-testid="stAlert"]:has([data-testid="stIconWarning"]),
.stAlert:has([data-testid="stIconWarning"]) {background:#fff6e6 !important;border-color:#f0d9a9 !important}
[data-testid="stDownloadButton"] button {background:#14263f !important;color:#ffffff !important;border:0 !important}
[data-testid="stDownloadButton"] button * {color:#ffffff !important;-webkit-text-fill-color:#ffffff !important}
.result-heading {font-family:'Manrope',sans-serif;color:var(--ink);font-weight:800}
.memory-item {background:#f1f8f7;border:1px solid #d8eeea;border-radius:11px;padding:.7rem .85rem;margin:.4rem 0;color:#244c4a;font-size:.9rem}
hr {border-color:#dfe8eb}

/* Deterministic app-owned notices: no theme-dependent Streamlit alert colours. */
.notice {display:block !important;padding:.9rem 1rem !important;margin:.65rem 0 !important;border-radius:12px !important;border:1px solid transparent !important;font-size:.94rem !important;line-height:1.55 !important;font-weight:600 !important;opacity:1 !important;}
.notice, .notice * {opacity:1 !important;}
.notice.success {background:#e7f5f1 !important;border-color:#b8e4d8 !important;color:#075f53 !important;-webkit-text-fill-color:#075f53 !important;}
.notice.info {background:#edf3fb !important;border-color:#d4e1f1 !important;color:#24466c !important;-webkit-text-fill-color:#24466c !important;}
.notice.warning {background:#fff6e6 !important;border-color:#f0d9a9 !important;color:#765016 !important;-webkit-text-fill-color:#765016 !important;}
.notice.error {background:#fff0ef !important;border-color:#efc5c1 !important;color:#8b2922 !important;-webkit-text-fill-color:#8b2922 !important;}
.notice.success *, .notice.info *, .notice.warning *, .notice.error * {color:inherit !important;-webkit-text-fill-color:inherit !important;}
.stButton button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {color:#fff !important;-webkit-text-fill-color:#fff !important;background:#14263f !important;border:1px solid #14263f !important;border-radius:10px !important;font-weight:700 !important;}
.stButton button[kind="primary"] {background:linear-gradient(90deg,#138f88,#20aa9b) !important;border:0 !important;}
.stButton button *, [data-testid="stDownloadButton"] button *, [data-testid="stFormSubmitButton"] button * {color:#fff !important;-webkit-text-fill-color:#fff !important;}
.stButton button:disabled {background:#dce5eb !important;color:#687b96 !important;-webkit-text-fill-color:#687b96 !important;border-color:#dce5eb !important;}

</style>
""", unsafe_allow_html=True)


def show_notice(message, kind="info"):
    """Render a high-contrast, theme-independent status message."""
    safe_message = html.escape(str(message)).replace("\n", "<br>")
    st.markdown(f'<div class="notice {kind}">{safe_message}</div>', unsafe_allow_html=True)


if "essay_result" not in st.session_state:
    st.session_state.essay_result = None
if "essay_topic" not in st.session_state:
    st.session_state.essay_topic = ""
if "memory_used" not in st.session_state:
    st.session_state.memory_used = []
if "memory_notice" not in st.session_state:
    st.session_state.memory_notice = ""

# -----------------------------
# Sidebar: workflow + memory
# -----------------------------
with st.sidebar:
    st.markdown("<div style='font-size:2rem'>✍️</div><h2 style='margin:.2rem 0'>EssayForge</h2><p style='color:#c1cfdb;margin-top:0'>Your local AI writing studio</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**YOUR WORKFLOW**")
    st.markdown('<div class="step">① <b>Draft Agent</b><br><small>Creates a first draft from your topic</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="step">② <b>Critique Agent</b><br><small>Checks clarity, structure and relevance</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="step">③ <b>Revision Agent</b><br><small>Improves the draft when issues are found</small></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🧠 WRITING MEMORY**")
    st.caption("Save writing preferences. EssayForge retrieves relevant memories from ChromaDB when drafting.")
    with st.form("save_memory_form", clear_on_submit=True):
        new_memory = st.text_input(
            "A preference to remember",
            placeholder="e.g. Use simple language and short paragraphs",
        )
        save_clicked = st.form_submit_button("＋ Save preference", use_container_width=True)
        if save_clicked:
            if new_memory.strip():
                try:
                    save_memory(new_memory.strip(), f"ui_memory_{uuid.uuid4()}")
                    st.session_state.memory_notice = "Preference saved to ChromaDB."
                except Exception as exc:
                    st.session_state.memory_notice = f"Could not save memory: {exc}"
            else:
                st.session_state.memory_notice = "Enter a preference before saving."
    if st.session_state.memory_notice:
        st.caption(st.session_state.memory_notice)

    try:
        saved = collection.get(include=["documents"])
        saved_docs = saved.get("documents") or []
        st.caption(f"{len(saved_docs)} saved writing preference(s)")
        if saved_docs:
            with st.expander("View saved preferences"):
                for item in saved_docs:
                    st.markdown(f'<div class="memory-item">{item}</div>', unsafe_allow_html=True)
    except Exception:
        st.caption("Memory store will be available when ChromaDB is ready.")


    st.markdown("---")
    st.markdown("**TECHNOLOGY**")
    st.caption("LangGraph · Ollama · ChromaDB · SQLite")
    try:
        health = requests.get("http://localhost:11434/api/tags", timeout=2)
        if health.ok:
            show_notice("Local AI service connected", "success")
        else:
            show_notice("Ollama service returned an error", "warning")
    except requests.RequestException:
        show_notice("Ollama is not reachable. Start Ollama first.", "warning")


# -----------------------------
# Main hero
# -----------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">AGENTIC AI • WRITING WORKSPACE</div>
  <h1>Write with clarity.<br>Improve with intelligence.</h1>
  <p>Turn a topic into a thoughtful paragraph through a coordinated draft, critique and revision workflow — shaped by your saved writing preferences.</p>
  <span class="pill">✦ Local AI&nbsp; · &nbsp;RAG memory&nbsp; · &nbsp;Guided feedback&nbsp; · &nbsp;One-click export</span>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.55, 1], gap="large")
with left:
    st.markdown('<div class="panel"><div class="eyebrow">01 / START WRITING</div><h3>What would you like to write about?</h3><p class="muted">Enter any clear topic or question. EssayForge creates a short student-level paragraph, reviews it and revises it when needed.</p></div>', unsafe_allow_html=True)
    topic = st.text_input(
        "Essay topic",
        placeholder="e.g. The role of artificial intelligence in education",
        label_visibility="collapsed",
        key="topic_input",
    )
    generate = st.button("✦ Generate and improve", type="primary", use_container_width=True, disabled=not topic.strip())
with right:
    st.markdown("""
    <div class="panel">
      <div class="eyebrow">02 / THE PROCESS</div><h3>One coordinated workflow</h3>
      <p class="muted">LangGraph controls the sequence and chooses the next step based on the critique.</p>
      <div class="step">📝 <b>Draft</b> — create the first version</div>
      <div class="step">🔎 <b>Critique</b> — identify writing issues</div>
      <div class="step">✨ <b>Revise or finish</b> — follow the conditional route</div>
      <div class="step">🧠 <b>RAG memory</b> — retrieve relevant saved preferences</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Run graph; show retrieved memory used by the drafting node
# -----------------------------
if generate:
    clean_topic = topic.strip()
    st.session_state.essay_result = None
    st.session_state.essay_topic = clean_topic
    st.session_state.memory_used = []
    memory_query = (
        f"Writing style, language, paragraph length, introduction and logical flow "
        f"for an essay about {clean_topic}"
    )
    try:
        st.session_state.memory_used = search_memory(memory_query)
    except Exception as exc:
        st.session_state.memory_used = []
        show_notice(f"Memory retrieval was unavailable for this run: {exc}", "warning")

    initial_state = {
        "messages": [],
        "topic": clean_topic,
        "draft": "",
        "critique": "",
        "issues_found": False,
        "revised_text": "",
        "demo_no_issues": False,
    }
    run_config = {"configurable": {"thread_id": f"essayforge-{uuid.uuid4()}"}}
    with st.spinner("Your agents are working... Your saved writing preferences are being retrieved."):
        try:
            st.session_state.essay_result = graph.invoke(initial_state, config=run_config)
        except Exception as exc:
            show_notice("The workflow could not complete. Check Ollama, the model and installed packages.", "error")
            st.exception(exc)

# -----------------------------
# Results
# -----------------------------
result = st.session_state.essay_result
if result:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">03 / YOUR OUTPUT</div><h2 class="result-heading">Your writing results</h2>', unsafe_allow_html=True)
    st.caption(f"Topic: {st.session_state.essay_topic}")

    with st.expander("🧠 Retrieved writing memory for this essay", expanded=True):
        if st.session_state.memory_used:
            show_notice(f"Retrieved {len(st.session_state.memory_used)} relevant preference(s) from ChromaDB.", "success")
            for memory in st.session_state.memory_used:
                st.markdown(f'<div class="memory-item">✓ {memory}</div>', unsafe_allow_html=True)
        else:
            show_notice("No matching saved preferences were retrieved. You can add preferences in the sidebar; the assistant can still write without them.", "info")

    draft = str(result.get("draft", "") or "").strip()
    critique = str(result.get("critique", "") or "").strip()
    revised = str(result.get("revised_text", "") or "").strip()
    if not draft:
        show_notice("The Draft Agent returned an empty result. Check essay_assistant.py and the Ollama model.", "error")
    else:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="panel"><div class="eyebrow">STAGE 1</div><h3>📝 Original draft</h3></div>', unsafe_allow_html=True)
            st.text_area("Draft text", value=draft, height=220, key="draft_output")
        with c2:
            st.markdown('<div class="panel"><div class="eyebrow">STAGE 2</div><h3>🔎 Critique</h3></div>', unsafe_allow_html=True)
            if critique.upper() == "NO_ISSUES":
                show_notice("✓ No significant writing issues were identified by the critique stage.", "success")
            else:
                show_notice(critique or "No critique was returned.", "info")

        st.markdown('<div class="panel"><div class="eyebrow">STAGE 3</div><h3>✨ Final result</h3></div>', unsafe_allow_html=True)
        if revised:
            show_notice("✓ Conditional routing selected the Revision Agent. Your draft has been revised.", "success")
            final_text = revised
        else:
            show_notice("The workflow finished without a revision step. The original draft is shown as the final result.", "info")
            final_text = draft
        st.text_area("Final paragraph", value=final_text, height=190, key="final_output")
        st.download_button("⬇️ Download final paragraph", data=final_text, file_name="essayforge_final_paragraph.txt", mime="text/plain", use_container_width=True)

    with st.expander("Technical run details"):
        st.write("Topic passed to graph:", st.session_state.essay_topic)
        st.write("Returned graph state keys:", list(result.keys()))
        st.caption("The Draft Agent independently calls search_memory() in rag_memory.py. The memory panel above previews the same retrieval query so the user can see which saved preferences are relevant to this essay.")

if st.button("＋ Start a new essay"):
    st.session_state.essay_result = None
    st.session_state.essay_topic = ""
    st.session_state.memory_used = []
    st.rerun()

st.markdown("<p style='text-align:center;color:#8190a4;font-size:.82rem;margin-top:2.5rem'>EssayForge · Local LangGraph + Ollama workflow · ChromaDB writing memory</p>", unsafe_allow_html=True)
