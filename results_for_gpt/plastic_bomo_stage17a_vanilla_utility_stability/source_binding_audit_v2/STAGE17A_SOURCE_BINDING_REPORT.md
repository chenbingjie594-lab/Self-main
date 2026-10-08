# Stage17A source binding audit

The rebuilt baseline passed its runtime smoke test. Formal 80-slot generation remains unauthorized.

Train sources: {'flash': 76, 'black': 36}. Binding counts: {'flash': {'NO_EXACT_CONTENT_BINDING': 76}, 'black': {'NO_EXACT_CONTENT_BINDING': 36}}.

Generation images are numbered 512x512 sources, whereas the frozen detector set contains full images. Only file/RGB equality was tested here, not visual nearest-neighbour assignment. Candidate bbox-crop filenames alone do not prove their transform or parent identity. An unbound source does not prove validation leakage.

Before formal slots, recover a verifiable source-to-real-parent and annotation/mask transform mapping. Do not silently create new crops, retrain a different pool, reorder numbered sources, select by quality, or replace RealRepeat with guessed parents. No historical statuses or labels were changed.
