STATUS_CHOICES = [
    ('pending', 'Pending'),
    ('processing', 'Processing'),
    ('completed', 'Completed'),
    ('failed', 'Failed'),
]

# ---------------------------------------------------------------------------
# Effect keys
# ---------------------------------------------------------------------------

# Motion
ANIMATION_FLOAT = 'float'
ANIMATION_BREATHE = 'breathe'
ANIMATION_NATURAL_BREATHE = 'natural-breathe'
ANIMATION_ZOOM = 'zoom'
ANIMATION_PULSE = 'pulse'
ANIMATION_TILT = 'tilt'

# Lighting
ANIMATION_GLOW = 'glow'
ANIMATION_RIM = 'rim'
ANIMATION_SHINE = 'shine'
ANIMATION_SHEEN = 'sheen'
ANIMATION_HALO = 'halo'
ANIMATION_GOLD_PULSE = 'gold_pulse'
ANIMATION_NEON_PULSE = 'neon_pulse'

# Overlay
ANIMATION_TWINKLE = 'twinkle'

# Legacy keys — still recognised so old jobs do not error, then stripped on save.
ANIMATION_BOUNCE = 'bounce'
ANIMATION_SHAKE = 'shake'
ANIMATION_WAVE = 'wave'
ANIMATION_SPIN = 'spin'
ANIMATION_FLOAT_GLOW = 'float-glow'
ANIMATION_SLIDE_UP = 'slide-up'
ANIMATION_SLIDE_LEFT = 'slide-left'
ANIMATION_ZOOM_IN = 'zoom-in'
ANIMATION_FLICKER = 'flicker'
ANIMATION_FADE = 'fade'
ANIMATION_RAINBOW = 'rainbow'
ANIMATION_SPARKLE = 'sparkle'
ANIMATION_COLOR_SHIFT = 'color_shift'

DEPRECATED_EFFECTS = frozenset({
    ANIMATION_BOUNCE,
    ANIMATION_SHAKE,
    ANIMATION_WAVE,
    ANIMATION_SPIN,
    ANIMATION_FLOAT_GLOW,
    ANIMATION_SLIDE_UP,
    ANIMATION_SLIDE_LEFT,
    ANIMATION_ZOOM_IN,
    ANIMATION_FLICKER,
    ANIMATION_FADE,
    ANIMATION_RAINBOW,
    ANIMATION_SPARKLE,
    ANIMATION_COLOR_SHIFT,
})

# ---------------------------------------------------------------------------
# Grouped choices (picker + validation)
# ---------------------------------------------------------------------------

EFFECT_GROUPS = [
    ('Person', [
        (ANIMATION_BREATHE, 'Breathe (Expand)'),
        (ANIMATION_NATURAL_BREATHE, 'Natural Breathe'),
        (ANIMATION_GLOW, 'Glow'),
        (ANIMATION_RIM, 'Rim light'),
        (ANIMATION_HALO, 'Back halo'),
        (ANIMATION_SHINE, 'Shine sweep'),
    ]),
    ('Card / UI', [
        (ANIMATION_ZOOM, 'Slow zoom'),
        (ANIMATION_PULSE, 'CTA pulse'),
        (ANIMATION_FLOAT, 'Idle float'),
        (ANIMATION_SHINE, 'Shine sweep'),
        (ANIMATION_SHEEN, 'Text sheen'),
        (ANIMATION_GLOW, 'Glow'),
        (ANIMATION_RIM, 'Rim light'),
        (ANIMATION_HALO, 'Back halo'),
        (ANIMATION_GOLD_PULSE, 'Gold pulse'),
        (ANIMATION_NEON_PULSE, 'Neon pulse'),
        (ANIMATION_TWINKLE, 'Corner twinkle'),
    ]),
    ('Prop', [
        (ANIMATION_FLOAT, 'Idle float'),
        (ANIMATION_TILT, 'Soft tilt'),
        (ANIMATION_SHINE, 'Shine sweep'),
        (ANIMATION_SHEEN, 'Text sheen'),
        (ANIMATION_GLOW, 'Glow'),
        (ANIMATION_HALO, 'Back halo'),
        (ANIMATION_GOLD_PULSE, 'Gold pulse'),
        (ANIMATION_TWINKLE, 'Corner twinkle'),
    ]),
]

ANIMATION_TYPE_CHOICES = [
    (ANIMATION_BREATHE, 'Breathe (Expand)'),
    (ANIMATION_NATURAL_BREATHE, 'Natural Breathe'),
    (ANIMATION_ZOOM, 'Slow zoom'),
    (ANIMATION_PULSE, 'CTA pulse'),
    (ANIMATION_FLOAT, 'Idle float'),
    (ANIMATION_TILT, 'Soft tilt'),
    (ANIMATION_SHINE, 'Shine sweep'),
    (ANIMATION_SHEEN, 'Text sheen'),
    (ANIMATION_GLOW, 'Glow'),
    (ANIMATION_RIM, 'Rim light'),
    (ANIMATION_HALO, 'Back halo'),
    (ANIMATION_GOLD_PULSE, 'Gold pulse'),
    (ANIMATION_NEON_PULSE, 'Neon pulse'),
    (ANIMATION_TWINKLE, 'Corner twinkle'),
]

ANIMATION_TYPE_LABELS = {
    **dict(ANIMATION_TYPE_CHOICES),
    ANIMATION_BOUNCE: 'Bounce',
    ANIMATION_SHAKE: 'Shake',
    ANIMATION_WAVE: 'Wave sway',
    ANIMATION_SPIN: 'Spin',
    ANIMATION_FLOAT_GLOW: 'Float & Glow',
    ANIMATION_SLIDE_UP: 'Slide Up',
    ANIMATION_SLIDE_LEFT: 'Slide from Left',
    ANIMATION_ZOOM_IN: 'Elastic Zoom',
    ANIMATION_FLICKER: 'Flicker',
    ANIMATION_FADE: 'Fade pulse',
    ANIMATION_RAINBOW: 'Rainbow Cycle',
    ANIMATION_SPARKLE: 'Sparkle',
    ANIMATION_COLOR_SHIFT: 'Color shift',
}

DEFAULT_ANIMATION_TYPES = [ANIMATION_SHINE]

EFFECT_APPLY_ORDER = [
    ANIMATION_PULSE,
    ANIMATION_ZOOM,
    ANIMATION_BREATHE,
    ANIMATION_NATURAL_BREATHE,
    ANIMATION_TILT,
    ANIMATION_FLOAT,
    ANIMATION_HALO,
    ANIMATION_GLOW,
    ANIMATION_GOLD_PULSE,
    ANIMATION_NEON_PULSE,
    ANIMATION_SHINE,
    ANIMATION_SHEEN,
    ANIMATION_RIM,
    ANIMATION_TWINKLE,
]
