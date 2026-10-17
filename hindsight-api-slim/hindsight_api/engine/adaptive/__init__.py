"""Adaptive memory: per-memory retention, reinforcement and claim state.

Three axes that never derive from one another (DESIGN.md §1): what enters a recall's context,
what is believed (claim state), and how readily a memory comes back without a strong cue
(retention). This package holds the pure parts — models, settings, the retention math and the
recall scoring hooks. Storage is the memories store's (``MemoriesExtension.load_adaptive_state``
and friends); nothing here names a table.

``adaptive_memory_mode=off`` (the default) must leave recall byte-identical, so every caller
resolves :class:`~hindsight_api.engine.adaptive.settings.AdaptiveSettings` first and skips all of
this when the effective mode is off.
"""
