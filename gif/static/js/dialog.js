(function () {
    const root = document.getElementById('app-dialog');
    if (!root) {
        window.showAppDialog = function (title, message) {
            window.alert(message || title);
        };
        return;
    }

    const titleNode = document.getElementById('app-dialog-title');
    const messageNode = document.getElementById('app-dialog-message');
    const okButton = document.getElementById('app-dialog-ok');
    let lastFocus = null;

    function closeDialog() {
        root.hidden = true;
        document.body.classList.remove('dialog-open');
        if (lastFocus && typeof lastFocus.focus === 'function') {
            lastFocus.focus();
        }
    }

    function showAppDialog(title, message) {
        lastFocus = document.activeElement;
        titleNode.textContent = title || 'Cannot continue';
        messageNode.textContent = message || '';
        root.hidden = false;
        document.body.classList.add('dialog-open');
        okButton.focus();
    }

    okButton.addEventListener('click', closeDialog);
    root.addEventListener('click', function (event) {
        if (event.target === root) {
            closeDialog();
        }
    });
    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape' && !root.hidden) {
            closeDialog();
        }
    });

    window.showAppDialog = showAppDialog;

    function selectedCount(form) {
        const ids = (form.querySelector('#detection-ids') || {}).value || '';
        const raw = (form.querySelector('#manual-regions') || {}).value || '[]';
        let extra = 0;
        try {
            const parsed = JSON.parse(raw);
            extra = Array.isArray(parsed) ? parsed.length : 0;
        } catch (err) {
            extra = 0;
        }
        const idCount = ids.split(',').map(function (part) {
            return part.trim();
        }).filter(Boolean).length;
        return idCount + extra;
    }

    document.addEventListener('submit', function (event) {
        const form = event.target;
        if (!(form instanceof HTMLFormElement)) {
            return;
        }

        if (form.id === 'animate-form') {
            if (selectedCount(form) === 0) {
                event.preventDefault();
                showAppDialog(
                    'Nothing selected',
                    'Select at least one title, card, button, or text box, or draw a region, before adjusting.'
                );
            }
            return;
        }

        if (form.id === 'adjust-form') {
            let regions = [];
            try {
                regions = JSON.parse((form.querySelector('#regions-input') || {}).value || '[]');
            } catch (err) {
                regions = [];
            }
            const effects = form.querySelectorAll('input[name="animation_types"]:checked');
            if (!regions.length) {
                event.preventDefault();
                showAppDialog(
                    'No regions left',
                    'Keep at least one region on the image. The GIF needs a box to animate.'
                );
                return;
            }
            if (!effects.length) {
                event.preventDefault();
                showAppDialog(
                    'No effect selected',
                    'Tick at least one animation effect before generating the GIF.'
                );
            }
            return;
        }

        if (form.id === 'upload-form') {
            const fileInput = form.querySelector('input[type="file"]');
            if (!fileInput || !fileInput.files || !fileInput.files.length) {
                event.preventDefault();
                showAppDialog(
                    'No image selected',
                    'Choose an image, or drop one onto the upload area, before continuing.'
                );
            }
        }
    });
})();
