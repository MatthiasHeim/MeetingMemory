-- Migration: Create meeting_metadata table
-- Purpose: 1:1 child of sources (where source_type='meeting') holding audio-only
-- signals that text alone cannot recover: overall sentiment, per-speaker emotional
-- arcs, pacing (wpm/hesitations/pauses), interruption events, energy levels.
--
-- Background: the prior pipeline asked Gemini for sentiment + lailix_feedback +
-- summaries inside the same prompt as transcription. For long meetings this was
-- silently broken (chunk-merge kept only chunk-1's analysis) and the structured
-- output never reached the DB anyway. This split moves audio-derived signal here
-- (irreplaceable from text) and pushes text-reasoning fields back to Claude with
-- full Brain context.

CREATE TABLE IF NOT EXISTS meeting_metadata (
  source_id INTEGER PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,

  -- Whole-meeting audio signal (post reduce-pass for chunked recordings)
  overall_sentiment TEXT,                -- positive | neutral | negative | mixed
  sentiment_intensity TEXT,              -- mild | moderate | strong

  -- Per-speaker affect (irreplaceable from transcript text alone)
  speaker_emotions JSONB,                -- [{speaker, arc:[{time, tone, energy, trigger, quote}]}]
  speaker_pacing JSONB,                  -- {speaker: {wpm_avg, hesitation_count, longest_pause_sec}}
  interruptions JSONB,                   -- [{time, interrupter, interruptee, context_quote}]
  energy_levels JSONB,                   -- {speaker: {avg, arc:[{time, level}]}}

  -- Processing metadata
  gemini_model TEXT,
  gemini_input_tokens INTEGER,
  gemini_output_tokens INTEGER,
  chunked BOOLEAN DEFAULT FALSE,
  chunk_count INTEGER,
  reduce_pass_used BOOLEAN DEFAULT FALSE,

  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_meeting_metadata_overall_sentiment
  ON meeting_metadata(overall_sentiment);

COMMENT ON TABLE meeting_metadata IS
  '1:1 child of sources where source_type=meeting. Holds audio-only signals (sentiment, emotions, pacing, interruptions) that text alone cannot recover.';
