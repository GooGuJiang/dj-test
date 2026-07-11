# Test report — Auto DJ 1.2.19 automatic queue resort

## Automated tests

- `pytest -q`: **98 passed**
- New regression coverage:
  - BPM/structure fallback reorders a queue without MuQ profiles.
  - Active transition prefix remains fixed during reordering.
  - Analysis completion invokes fallback sorting when MuQ is disabled.
  - Automatic sort requests are retained while another analysis phase is active.

## Offline audio checks

- `python check_natural_transition.py`: PASS
- `python check_mixend_continuity.py`: PASS
- `python check_drum_loop_bridge.py`: PASS
