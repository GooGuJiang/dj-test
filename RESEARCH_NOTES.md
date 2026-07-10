# Research notes

All-In-One predicts tempo, beats, downbeats, functional segment boundaries and labels. The application keeps Beat This! as the timing authority and samples All-In-One labels on the Beat This! downbeat grid.

All-In-One labels are fused with local energy, onset density, bass ratio, vocal proxy and novelty features. DJ-oriented roles such as BUILDUP, DROP and BREAKDOWN are fusion outputs, not direct All-In-One labels.

Version 1.2.4 runs All-In-One in a background thread inside the GUI process. A module-level lock serializes calls. `multiprocess=False` is used for spectrogram extraction, and CPU is the default structure-analysis device to protect GPU memory for MuQ, Beat This! and real-time preload.
