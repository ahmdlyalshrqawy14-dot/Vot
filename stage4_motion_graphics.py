import math
from typing import Dict, Any, List, Optional
from pathlib import Path

def generate_accent_bar_filter(start: float, end: float, y_pos: str = "H-200") -> str:
    """Generates a filter string for a subtle colored accent bar/underline sliding in."""
    dur = end - start
    if dur <= 0:
        return ""

    color = "0x47E0FD" # Yellowish/cyan accent
    # Draw a thin box that scales horizontally based on time (t-start).
    # Width grows from 0 to W/3 over 0.3s.
    w_expr = f"min(W/3, (t-{start})/0.3 * W/3)"
    x_expr = "(W-w)/2" # Centered

    # We use drawbox on the existing video stream. Since we are building complex graphs,
    # we usually apply this to the current video stream.
    # format: drawbox=x=expr:y=expr:w=expr:h=8:color=color:t=fill:enable='between(t,start,end)'
    return f"drawbox=x='{x_expr}':y='{y_pos}':w='{w_expr}':h=8:color={color}@0.8:t=fill:enable='between(t,{start},{end})'"

def generate_lower_third_label_filter(start: float, end: float, keyword: str) -> str:
    """Generates a text label (lower third) overlay for 'myth' or 'definition' roles."""
    if not keyword:
        return ""

    dur = end - start
    if dur <= 0:
        return ""

    # We use drawtext.
    # We add a background box to the text.
    # We fade it in and out.
    safe_kw = keyword.replace("'", "").replace(":", "")
    fade_in = f"alpha='if(lt(t,{start+0.3}), (t-{start})/0.3, if(gt(t,{end-0.3}), ({end}-t)/0.3, 1))'"

    return f"drawtext=text='{safe_kw}':fontcolor=white:fontsize=48:box=1:boxcolor=black@0.6:boxborderw=10:x=100:y=H-150:enable='between(t,{start},{end})':{fade_in}"

def generate_focus_vignette_filter(start: float, end: float, intensity: float) -> str:
    """Generates a subtle vignette pulse for high intensity shots."""
    dur = end - start
    if dur <= 0 or intensity < 0.8:
        return ""

    # Vignette filter: vignette=PI/4
    # We use enable to only apply it during the specified time.
    # angle depends on intensity, max PI/3
    angle = min(math.pi / 3, math.pi / 5 * intensity)
    return f"vignette=a='{angle}':enable='between(t,{start},{end})'"
