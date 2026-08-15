-- Schema for the "accidents" table (SkyRAG project)

CREATE TABLE IF NOT EXISTS accidents (
    ntsb_no                 TEXT PRIMARY KEY,
    event_id                TEXT,
    event_date              TIMESTAMP,
    city                    TEXT,
    state                   TEXT,
    country                 TEXT,
    latitude                DOUBLE PRECISION,
    longitude               DOUBLE PRECISION,

    make                    TEXT,
    model                   TEXT,
    aircraft_category       TEXT,
    number_of_engines       TEXT,
    engine_type             TEXT,
    amateur_built           TEXT,

    operator                TEXT,
    purpose_of_flight       TEXT,
    scheduled               TEXT,
    far                     TEXT,

    fatal_injury_count      INTEGER DEFAULT 0,
    serious_injury_count    INTEGER DEFAULT 0,
    minor_injury_count      INTEGER DEFAULT 0,
    onboard_injury_count    INTEGER DEFAULT 0,
    on_ground_injury_count  INTEGER DEFAULT 0,
    highest_injury_level    TEXT,

    aircraft_damage         TEXT,
    weather_condition       TEXT,
    broad_phase_of_flight   TEXT,

    report_status           TEXT,
    report_url              TEXT,
    docket_url               TEXT
);

-- Indexes for the queries the dashboard/agent will run most often
CREATE INDEX IF NOT EXISTS idx_accidents_event_date ON accidents (event_date);
CREATE INDEX IF NOT EXISTS idx_accidents_state ON accidents (state);
CREATE INDEX IF NOT EXISTS idx_accidents_make ON accidents (make);