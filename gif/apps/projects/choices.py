SOURCE_YOLO = 'yolo'
SOURCE_OCR = 'ocr'
SOURCE_MANUAL = 'manual'
SOURCE_CARD = 'card'
SOURCE_BUTTON = 'button'
SOURCE_TITLE = 'title'
SOURCE_PROP = 'prop'

SOURCE_CHOICES = [
    (SOURCE_YOLO, 'Yolo'),
    (SOURCE_OCR, 'OCR'),
    (SOURCE_MANUAL, 'Manual'),
    (SOURCE_CARD, 'Card'),
    (SOURCE_BUTTON, 'Button'),
    (SOURCE_TITLE, 'Title'),
    (SOURCE_PROP, 'Prop'),
]

# Auto-detected rows are replaced on every Re-run. Manual draws are kept.
AUTO_DETECTION_SOURCES = (
    SOURCE_YOLO,
    SOURCE_OCR,
    SOURCE_CARD,
    SOURCE_BUTTON,
    SOURCE_TITLE,
    SOURCE_PROP,
)
