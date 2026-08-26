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
ANIMATION_ZOOM = 'zoom'
ANIMATION_BOUNCE = 'bounce'
ANIMATION_SHAKE = 'shake'
ANIMATION_WAVE = 'wave'
ANIMATION_SPIN = 'spin'

# Lighting
ANIMATION_GLOW = 'glow'
ANIMATION_RIM = 'rim'
ANIMATION_SHINE = 'shine'
ANIMATION_GOLD_PULSE = 'gold_pulse'
ANIMATION_FLICKER = 'flicker'
ANIMATION_FADE = 'fade'

# Particle / colour
ANIMATION_SPARKLE = 'sparkle'
ANIMATION_COLOR_SHIFT = 'color_shift'

# ---------------------------------------------------------------------------
# Grouped choices (used by the template effect picker)
# ---------------------------------------------------------------------------

EFFECT_GROUPS = [
    ('Motion', [
        (ANIMATION_FLOAT,   'Idle float'),
        (ANIMATION_BREATHE, 'Breathe'),
        (ANIMATION_ZOOM,    'Slow zoom'),
        (ANIMATION_BOUNCE,  'Bounce'),
        (ANIMATION_SHAKE,   'Shake'),
        (ANIMATION_WAVE,    'Wave sway'),
        (ANIMATION_SPIN,    'Spin'),
    ]),
    ('Lighting', [
        (ANIMATION_GLOW,       'Glow pulse'),
        (ANIMATION_RIM,        'Rim light'),
        (ANIMATION_SHINE,      'Shine sweep'),
        (ANIMATION_GOLD_PULSE, 'Gold pulse'),
        (ANIMATION_FLICKER,    'Flicker'),
        (ANIMATION_FADE,       'Fade pulse'),
    ]),
    ('Particle', [
        (ANIMATION_SPARKLE,     'Sparkle'),
        (ANIMATION_COLOR_SHIFT, 'Color shift'),
    ]),
]

# Flat list kept for model choices field and validation
ANIMATION_TYPE_CHOICES = [
    (value, label)
    for _group, items in EFFECT_GROUPS
    for value, label in items
]

ANIMATION_TYPE_LABELS = dict(ANIMATION_TYPE_CHOICES)

DEFAULT_ANIMATION_TYPES = [ANIMATION_SHINE]

# Motion effects run first so lighting composites onto moving pixels.
EFFECT_APPLY_ORDER = [
    ANIMATION_ZOOM,
    ANIMATION_BREATHE,
    ANIMATION_BOUNCE,
    ANIMATION_SHAKE,
    ANIMATION_WAVE,
    ANIMATION_SPIN,
    ANIMATION_FLOAT,
    ANIMATION_FADE,
    ANIMATION_GLOW,
    ANIMATION_GOLD_PULSE,
    ANIMATION_FLICKER,
    ANIMATION_SHINE,
    ANIMATION_RIM,
    ANIMATION_SPARKLE,
    ANIMATION_COLOR_SHIFT,
]
