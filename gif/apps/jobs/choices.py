STATUS_CHOICES = [
    ('pending', 'Pending'),
    ('processing', 'Processing'),
    ('completed', 'Completed'),
    ('failed', 'Failed'),
]

ANIMATION_SHINE = 'shine'
ANIMATION_GLOW = 'glow'
ANIMATION_GOLD_PULSE = 'gold_pulse'
ANIMATION_BREATHE = 'breathe'
ANIMATION_FLOAT = 'float'
ANIMATION_SPARKLE = 'sparkle'
ANIMATION_FADE = 'fade'
ANIMATION_ZOOM = 'zoom'
ANIMATION_RIM = 'rim'
ANIMATION_FLICKER = 'flicker'

ANIMATION_TYPE_CHOICES = [
    (ANIMATION_SHINE, 'Gold shine sweep'),
    (ANIMATION_GLOW, 'Soft glow'),
    (ANIMATION_GOLD_PULSE, 'Gold colour pulse'),
    (ANIMATION_BREATHE, 'Subtle breathe'),
    (ANIMATION_FLOAT, 'Idle float'),
    (ANIMATION_SPARKLE, 'Sparkle'),
    (ANIMATION_FADE, 'Fade pulse'),
    (ANIMATION_ZOOM, 'Slow zoom'),
    (ANIMATION_RIM, 'Rim light'),
    (ANIMATION_FLICKER, 'Gold flicker'),
]

DEFAULT_ANIMATION_TYPE = ANIMATION_SHINE
DEFAULT_ANIMATION_TYPES = [ANIMATION_SHINE]

# Motion first so lighting/overlays run on the moving pixels.
EFFECT_APPLY_ORDER = [
    ANIMATION_ZOOM,
    ANIMATION_BREATHE,
    ANIMATION_FLOAT,
    ANIMATION_FADE,
    ANIMATION_GLOW,
    ANIMATION_GOLD_PULSE,
    ANIMATION_FLICKER,
    ANIMATION_SHINE,
    ANIMATION_RIM,
    ANIMATION_SPARKLE,
]
