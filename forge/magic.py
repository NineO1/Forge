"""
Forge Magic Button — Phase 1 (rule-based prompt enhancer).

Rewrites a plain user prompt into a cinematic, motion-rich prompt
tailored for LTX-Video. Zero dependencies. Original text is always
preserved at the front of the output.
"""

ENHANCER_VERSION = "1.0"

# ---------------------------------------------------------------- term banks
# Ordered by prompt-adherence value for LTX-Video: motion and camera first,
# since LTX rewards early-position motion verbs over late-position adjectives.

CAMERA_MOVE = [
    "slow cinematic dolly-in",
    "smooth tracking shot following the subject",
    "gentle orbiting crane shot",
    "steady handheld camera with subtle sway",
]

SHOT_TYPE = [
    "medium close-up shot",
    "wide establishing shot",
    "low-angle dynamic shot",
    "over-the-shoulder shot",
]

LIGHTING = [
    "dramatic volumetric lighting",
    "soft golden-hour light",
    "moody rim lighting with deep shadows",
    "even cinematic key lighting",
]

ATMOSPHERE = [
    "fine atmospheric haze, dust motes drifting in the light",
    "subtle floating particles in the air",
    "mist rolling gently through the scene",
]

MOTION_QUALITY = [
    "fluid natural motion, smooth and continuous movement",
    "physically plausible slow-motion dynamics",
    "lively energetic motion with believable weight",
]

STYLE = [
    "cinematic film look, 35mm, shallow depth of field, film grain",
    "high-detail photorealistic render, professional color grading",
    "epic movie still composition, rule of thirds",
]

# Keywords that indicate a category is ALREADY handled by the user.
_HINTS = {
    "camera_move": ["camera", "shot", "pan", "dolly", "zoom", "tracking",
                    "crane", "orbit", "handheld"],
    "shot_type": ["close-up", "closeup", "wide shot", "establishing",
                  "low-angle", "aerial", "macro"],
    "lighting": ["light", "lighting", "lit", "golden hour", "backlit",
                 "neon", "shadow", "glow", "sunset", "sunrise"],
    "atmosphere": ["fog", "mist", "haze", "smoke", "particles", "dust",
                   "rain", "snow"],
    "motion_quality": ["slow motion", "slow-motion", "motion", "movement",
                       "moving", "walking", "running", "flowing"],
    "style": ["cinematic", "photorealistic", "film", "grain", "render",
              "anime", "cartoon", "painting"],
}


def _stable_pick(bank, text, salt):
    """Deterministically pick a term from a bank based on the prompt text."""
    h = sum(ord(c) for c in (salt + text.strip().lower()))
    return bank[h % len(bank)]


def _has_hint(prompt_lower, category):
    return any(h in prompt_lower for h in _HINTS[category])


def enhance_prompt(text: str) -> str:
    """
    Enhance a user prompt for LTX-Video generation.

    Returns the enhanced prompt. The user's original wording is always
    kept as the leading clause — enhancement adds cinematic scaffolding
    around it, never replaces it.
    """
    original = (text or "").strip()
    if not original:
        return ""

    low = original.lower()

    # Opening phrase gives LTX a narrative anchor.
    openers = [
        "The scene opens with",
        "Cinematic shot of",
        "In this cinematic sequence,",
    ]
    opener = _stable_pick(openers, original, salt="opener")

    # Strip trailing periods so clauses join cleanly.
    body = original.rstrip(".!? ")
    head = f"{opener} {body}"

    additions = []
    if not _has_hint(low, "camera_move"):
        additions.append(_stable_pick(CAMERA_MOVE, original, "cam"))
    if not _has_hint(low, "shot_type"):
        additions.append(_stable_pick(SHOT_TYPE, original, "shot"))
    if not _has_hint(low, "lighting"):
        additions.append(_stable_pick(LIGHTING, original, "light"))
    if not _has_hint(low, "atmosphere"):
        additions.append(_stable_pick(ATMOSPHERE, original, "atmo"))
    if not _has_hint(low, "motion_quality"):
        additions.append(_stable_pick(MOTION_QUALITY, original, "mot"))
    if not _has_hint(low, "style"):
        additions.append(_stable_pick(STYLE, original, "sty"))

    if not additions:
        # User already covered everything — add pacing guidance only.
        additions.append("paced deliberately, holding a single continuous shot")

    return head + ", " + ", ".join(additions) + "."


if __name__ == "__main__":
    import sys
    sample = " ".join(sys.argv[1:]) or "a man standing on a rooftop at night"
    print(enhance_prompt(sample))