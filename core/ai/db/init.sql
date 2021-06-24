CREATE TABLE IF NOT EXISTS genre_distributions
    (
    user_id NUMERIC NOT NULL PRIMARY KEY,
	user_data BYTEA
	);

CREATE TABLE IF NOT EXISTS GENRE_COMBS
    (
    center_genre NUMERIC NOT NULL,
    context_genre NUMERIC NOT NULL,
    rank NUMERIC,
    overlap NUMERIC,
    acoustic_factor NUMERIC,
    word_sim NUMERIC,
    region_sim NUMERIC,
    final_score NUMERIC
	);

CREATE TABLE IF NOT EXISTS GENRE_COMBS_LITE
    (
    CENTER_GENRE NUMERIC NOT NULL,
    CONTEXT_GENRE NUMERIC NOT NULL,
    SCORE NUMERIC
	);

CREATE UNIQUE INDEX IF NOT EXISTS combs_idx ON GENRE_COMBS (center_genre, context_genre);
CREATE UNIQUE INDEX IF NOT EXISTS combs_lite_idx ON GENRE_COMBS_LITE (center_genre, context_genre);