"""
Aggregate SQL queries used by the dashboard's Analytics tab. Each query
runs server-side (GROUP BY/COUNT in Postgres) rather than pulling all rows
into pandas, and results are cached by Streamlit to avoid re-querying
Supabase on every UI interaction.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        database_url = os.getenv("DATABASE_URL")
        _engine = create_engine(database_url)
    return _engine


@st.cache_data(ttl=3600)
def get_summary_stats() -> dict:
    query = """
        SELECT
            COUNT(*) AS total_accidents,
            SUM(fatal_injury_count) AS total_fatalities,
            MIN(event_date) AS earliest_date,
            MAX(event_date) AS latest_date
        FROM accidents;
    """
    with get_engine().connect() as conn:
        row = conn.execute(text(query)).mappings().one()
    return dict(row)


@st.cache_data(ttl=3600)
def get_accidents_by_year() -> pd.DataFrame:
    query = """
        SELECT EXTRACT(YEAR FROM event_date)::int AS year, COUNT(*) AS accidents
        FROM accidents
        WHERE event_date IS NOT NULL
        GROUP BY year
        ORDER BY year;
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn)


@st.cache_data(ttl=3600)
def get_top_states(limit: int = 10) -> pd.DataFrame:
    query = """
        SELECT state, COUNT(*) AS accidents
        FROM accidents
        WHERE state IS NOT NULL
        GROUP BY state
        ORDER BY accidents DESC
        LIMIT :limit;
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn, params={"limit": limit})


@st.cache_data(ttl=3600)
def get_top_makes(limit: int = 10) -> pd.DataFrame:
    query = """
        SELECT make, COUNT(*) AS accidents
        FROM accidents
        WHERE make IS NOT NULL
        GROUP BY make
        ORDER BY accidents DESC
        LIMIT :limit;
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn, params={"limit": limit})


@st.cache_data(ttl=3600)
def get_severity_distribution() -> pd.DataFrame:
    query = """
        SELECT
            CASE
                WHEN fatal_injury_count > 0 THEN 'Fatal'
                WHEN serious_injury_count > 0 THEN 'Serious'
                WHEN minor_injury_count > 0 THEN 'Minor'
                ELSE 'None'
            END AS severity,
            COUNT(*) AS accidents
        FROM accidents
        GROUP BY severity;
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn)


@st.cache_data(ttl=3600)
def get_weather_distribution() -> pd.DataFrame:
    query = """
        SELECT weather_condition, COUNT(*) AS accidents
        FROM accidents
        WHERE weather_condition IS NOT NULL
        GROUP BY weather_condition
        ORDER BY accidents DESC;
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn)
