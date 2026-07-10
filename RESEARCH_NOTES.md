# Research and implementation notes — 1.2.2

The runtime structure model is the official `ASLP-lab/SongFormer` checkpoint.
SongFormer boundaries and functional labels are snapped to the Beat This! downbeat grid before being used by the transition matcher.

MuQ provides global, intro, outro and local timeline embeddings. Playlist ordering uses directional Outro→Intro compatibility together with BPM, energy, timbre and structural-role penalties.

EDM roles such as BUILDUP, DROP and BREAKDOWN are engineering-level fusion outputs derived from SongFormer labels plus local energy, onset, bass and novelty features. They are not presented as direct SongFormer labels.

Version 1.2.1 removes the obsolete model setter that only accepted `harmonix-*` names. The public engine API is now exclusively `set_songformer_enabled`, `set_songformer_device`, `set_songformer_model` and `set_preloaded_songformer_profiles`.


## Analysis observability

Version 1.2.2 uses structured `AUTODJ_PROGRESS` JSON lines between the isolated SongFormer worker and the GUI process. Ordinary stdout/stderr is forwarded to the parent console while structured lines update the GUI progress bar. Beat This! and MuQ use in-process callbacks, with MuQ reporting fractional progress per semantic window.
