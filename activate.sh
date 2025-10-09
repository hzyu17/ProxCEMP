#!/bin/bash
# Quick activation script
source venv/bin/activate
echo "Virtual environment activated!"
echo "Python: $(which python)"
echo ""
echo "Available scripts:"
echo "  python motion_planner.py       - Plan trajectory"
echo "  python view_trajectory.py      - View in simulator"
echo "  python visualize_path.py       - Create animation"
echo "  python export_video.py         - Export to video"
