// Single source of truth for frontend constants.
// Backend has its own mirror in src/mada/interfaces/planner/plan_schema.py
// (see DEFAULT_MODEL there). Keep these in sync.

/** Default LLM model used across the planner UI. */
export const DEFAULT_MODEL = "gpt-4o-mini";

/**
 * Full list of selectable models for the Orchestration Settings dropdown.
 * Ordered by likely preference (default first).
 */
export const AVAILABLE_MODELS = [
  "gpt-4o-mini",
  "gpt-4o",
  "gpt-5",
  "o3",
  "claude-sonnet-4-6",
  "claude-opus-4-6",
] as const;
