"""
Loads accidents_structured.csv into the Supabase Postgres "accidents" table.

Requires a .env file at the project root with:
    DATABASE_URL=postgresql://postgres:PASSWORD@db.xxxx.supabase.co:5432/postgres
"""

import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "data" / "processed" / "accidents_structured.csv"
SCHEMA_PATH = BASE_DIR / "src" / "sql" / "schema.sql"

# Maps CSV columns (pandas/original names) -> SQL columns (snake_case, matching schema.sql)
COLUMN_MAPPING = {
    "NtsbNo": "ntsb_no",
    "EventID": "event_id",
    "EventDate": "event_date",
    "City": "city",
    "State": "state",
    "Country": "country",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Make": "make",
    "Model": "model",
    "AirCraftCategory": "aircraft_category",
    "NumberOfEngines": "number_of_engines",
    "EngineType": "engine_type",
    "AmateurBuilt": "amateur_built",
    "Operator": "operator",
    "PurposeOfFlight": "purpose_of_flight",
    "Scheduled": "scheduled",
    "FAR": "far",
    "FatalInjuryCount": "fatal_injury_count",
    "SeriousInjuryCount": "serious_injury_count",
    "MinorInjuryCount": "minor_injury_count",
    "OnboardInjuryCount": "onboard_injury_count",
    "OnGroundInjuryCount": "on_ground_injury_count",
    "HighestInjuryLevel": "highest_injury_level",
    "AirCraftDamage": "aircraft_damage",
    "WeatherCondition": "weather_condition",
    "BroadPhaseofFlight": "broad_phase_of_flight",
    "ReportStatus": "report_status",
    "ReportUrl": "report_url",
    "DocketUrl": "docket_url",
}


def get_engine():
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found. Check your .env file.")
    return create_engine(database_url)


def create_schema(engine) -> None:
    print("Creating table (if it doesn't already exist)...")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    with engine.begin() as conn:
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.exec_driver_sql(statement)
    print("Schema ready.")


def load_and_prepare_data() -> pd.DataFrame:
    print("Loading structured CSV...")
    df = pd.read_csv(CSV_PATH)

    # Keep only the columns we mapped, rename them to match the SQL schema
    available = [c for c in COLUMN_MAPPING if c in df.columns]
    df = df[available].rename(columns=COLUMN_MAPPING)

    print(f"{len(df)} rows ready to import, {len(df.columns)} columns.")
    return df


def import_to_supabase(df: pd.DataFrame, engine) -> None:
    print("Importing into Supabase (table: accidents)...")
    df.to_sql(
        "accidents",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
    print(f"Import complete: {len(df)} rows inserted.")


if __name__ == "__main__":
    engine = get_engine()
    create_schema(engine)
    data = load_and_prepare_data()
    import_to_supabase(data, engine)