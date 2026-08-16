"""
SkyRAG dashboard: chat interface over the hybrid agent (SQL + RAG),
plus analytics and evaluation tabs. Aero (sky) themed.
"""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from agent.hybrid_agent import ask
from eval.eval_history import load_run_history
from sql.analytics_queries import (
    get_accidents_by_year,
    get_severity_distribution,
    get_summary_stats,
    get_top_makes,
    get_top_states,
    get_weather_distribution,
)

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
    st.markdown("### Aviation Accident Analytics")
    st.caption("NTSB accident reports, 2016-2023 — from the Supabase database.")

    stats = get_summary_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total accidents", f"{stats['total_accidents']:,}")
    col2.metric("Total fatalities", f"{int(stats['total_fatalities']):,}")
    col3.metric("Earliest", str(stats["earliest_date"]))
    col4.metric("Latest", str(stats["latest_date"]))

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Accidents by year")
        df_year = get_accidents_by_year()
        fig = px.bar(df_year, x="year", y="accidents", color_discrete_sequence=["#2b7bab"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top 10 states")
        df_states = get_top_states()
        fig = px.bar(df_states, x="accidents", y="state", orientation="h",
                     color_discrete_sequence=["#1e5f8c"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Injury severity")
        df_severity = get_severity_distribution()
        fig = px.pie(df_severity, names="severity", values="accidents", hole=0.45,
                     color="severity",
                     color_discrete_map={"Fatal": "#c0392b", "Serious": "#e67e22",
                                          "Minor": "#f1c40f", "None": "#2ecc71"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top 10 aircraft manufacturers")
        df_makes = get_top_makes()
        fig = px.bar(df_makes, x="accidents", y="make", orientation="h",
                     color_discrete_sequence=["#5fa8d3"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Weather conditions")
    df_weather = get_weather_distribution()
    fig = px.bar(df_weather, x="weather_condition", y="accidents",
                 color_discrete_sequence=["#3498db"])
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tab_eval:
    st.markdown("### RAG Evaluation History")
    st.caption("RAGAS scores across evaluation runs, tracking the effect "
               "of each fix made to the retrieval pipeline.")

    history = load_run_history()

    if history.empty:
        st.info("No evaluation runs found yet. Run "
                "`python src/eval/run_evaluation.py` to create one.")
    else:
        history_display = history.copy()
        history_display["run"] = [f"Run {i + 1}" for i in range(len(history_display))]

        metric_cols = ["retrieval_hit_rate", "faithfulness",
                        "answer_relevancy", "context_precision", "context_recall"]
        available_metrics = [c for c in metric_cols if c in history_display.columns]

        df_melted = history_display.melt(
            id_vars="run", value_vars=available_metrics,
            var_name="metric", value_name="score",
        )

        fig = px.line(
            df_melted, x="run", y="score", color="metric", markers=True,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0, 1], yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Run details")
        display_cols = ["run", "run_id", "n_questions"] + available_metrics
        st.dataframe(
            history_display[display_cols].style.format(
                {c: "{:.1%}" for c in available_metrics}
            ),
            use_container_width=True,
        )
