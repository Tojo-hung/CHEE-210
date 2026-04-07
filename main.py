"""
main.py
Backward-compatible master entry point.

The project now exposes phase-specific runners:
    python run_phase1.py
    python run_phase2.py
    python run_phase3.py
    python run_all.py

main.py remains as a thin wrapper so existing habits still work.
"""

from run_all import main


if __name__ == "__main__":
    main()
