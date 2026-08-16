"""
SkyRAG dashboard: chat interface over the hybrid agent (SQL + RAG),
plus analytics and evaluation tabs. Aero (sky) themed.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from agent.hybrid_agent import ask

st.set_page_config(
    page_title="SkyRAG - Aviation Safety Assistant",
    page_icon="✈️",
    layout="wide",
)

# --- Aero theme ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #cfe8f7 0%, #eaf6fd 40%, #ffffff 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e5f8c 0%, #2b7bab 100%);
}
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
h1, h2, h3 {
    color: #0f3d5c;
}
.stChatMessage {
    background-color: rgba(255, 255, 255, 0.75);
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #1e5f8c;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ✈️ SkyRAG")
    st.markdown("Aviation safety assistant, powered by a hybrid "
                "SQL + RAG pipeline over NTSB accident reports (2016-2023).")
    st.markdown("---")
    st.markdown("**Ask about:**")
    st.markdown("- Why an accident happened (RAG)")
    st.markdown("- Statistics & counts (SQL)")
    st.markdown("- Both at once")

tab_chat, tab_analytics, tab_eval = st.tabs(
    ["💬 Chat", "📊 Analytics", "🧪 Evaluation"]
)

with tab_chat:
    st.markdown("### Ask SkyRAG")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("route"):
                st.caption(f"Route: {message['route']}")
            if message.get("sql_query"):
                with st.expander("SQL query used"):
                    st.code(message["sql_query"], language="sql")
            if message.get("sources"):
                st.caption("Sources: " + ", ".join(message["sources"]))

    if question := st.chat_input("Ask a question about aviation accidents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(question)
            st.markdown(result["answer"])
            st.caption(f"Route: {result['route']}")
            if result.get("sql_query"):
                with st.expander("SQL query used"):
                    st.code(result["sql_query"], language="sql")
            if result.get("sources"):
                st.caption("Sources: " + ", ".join(result["sources"]))

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "route": result["route"],
            "sql_query": result.get("sql_query"),
            "sources": result.get("sources"),
        })

with tab_analytics:
    st.markdown("### Coming next")
    st.info("This tab will show accident statistics from the SQL database "
            "(by year, state, aircraft, severity).")

with tab_eval:
    st.markdown("### Coming next")
    st.info("This tab will show RAGAS evaluation scores over time, "
            "tracking retrieval and generation quality across runs.")
