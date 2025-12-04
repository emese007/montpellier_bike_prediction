-- =========================================================
-- Schéma base de données : Montpellier Bike Prediction
-- =========================================================
-- Ordre de création :
-- 1) counters
-- 2) bike_hourly
-- 3) weather_hourly
-- 4) holidays
-- 5) weather_forecast_hourly
-- 6) bike_predictions_hourly
-- =========================================================

-- 1) Compteurs vélo (métadonnées fixes)
------------------------------------------------------------
create table if not exists counters (
    id   text primary key,        -- urn:ngsi-ld:EcoCounter:...
    name text,                    -- nom lisible (optionnel pour l'instant)
    lat  double precision,        -- latitude
    lon  double precision         -- longitude
    -- pas de created_at obligatoire ici, c'est très statique
);

-- 2) Mesures horaires de trafic vélo (historique réel)
------------------------------------------------------------
create table if not exists bike_hourly (
    counter_id    text not null references counters(id) on delete cascade,
    timestamp_utc timestamptz not null,
    intensity     integer not null,
    created_at    timestamptz default now(),
    primary key (counter_id, timestamp_utc)
);

-- Index pour les requêtes par date
create index if not exists idx_bike_hourly_timestamp
    on bike_hourly (timestamp_utc);

-- 3) Météo horaire observée (historique Open-Meteo)
------------------------------------------------------------
create table if not exists weather_hourly (
    timestamp_utc        timestamptz primary key,
    temperature_2m       double precision,
    relative_humidity_2m double precision,
    precipitation        double precision,
    wind_speed_10m       double precision,
    created_at           timestamptz default now()
);

-- 4) Jours fériés (France métropole, 2023 → année courante)
------------------------------------------------------------
create table if not exists holidays (
    date       date primary key,
    name       text not null,
    zone       text not null,
    year       integer not null,
    created_at timestamptz default now()
);

-- 5) Prévisions météo horaires (J+1, J+2, etc.)
------------------------------------------------------------
create table if not exists weather_forecast_hourly (
    timestamp_utc        timestamptz primary key,  -- heure future
    temperature_2m       double precision,
    relative_humidity_2m double precision,
    precipitation        double precision,
    wind_speed_10m       double precision,
    created_at           timestamptz default now()
);

-- 6) Prédictions horaires de trafic vélo (Prophet / XGBoost)
------------------------------------------------------------
create table if not exists bike_predictions_hourly (
    counter_id         text not null references counters(id) on delete cascade,
    timestamp_utc      timestamptz not null,          -- heure prédite
    predicted_intensity double precision not null,    -- valeur prédite
    model_name         text not null,                 -- "prophet", "xgboost", etc.
    created_at         timestamptz default now(),
    primary key (counter_id, timestamp_utc, model_name)
);

create index if not exists idx_bike_predictions_hourly_timestamp
    on bike_predictions_hourly (timestamp_utc);
