(function () {
    const image = document.getElementById('source-image');
    const canvas = document.getElementById('overlay');
    const wrap = document.getElementById('canvas-wrap');
    const dataNode = document.getElementById('regions-data');
    const regionsInput = document.getElementById('regions-input');
    const form = document.getElementById('adjust-form');
    const list = document.getElementById('region-list');
    const effectPanel = document.getElementById('effect-panel');

    if (!image || !canvas || !dataNode) {
        return;
    }

    const HANDLE = 8;
    const MIN = 0.02;
    const cursors = {
        nw: 'nwse-resize', n: 'ns-resize', ne: 'nesw-resize',
        e: 'ew-resize', se: 'nwse-resize', s: 'ns-resize',
        sw: 'nesw-resize', w: 'ew-resize', move: 'move',
    };

    // -----------------------------------------------------------------------
    // Effect groups — kept in sync with choices.py
    // -----------------------------------------------------------------------
    const EFFECT_GROUPS = [
        {
            label: 'Animation',
            effects: [
                { key: 'float', label: 'Idle float' },
                { key: 'float-glow', label: 'Float & Glow' },
                { key: 'breathe', label: 'Breathe' },
                { key: 'zoom', label: 'Slow zoom' },
                { key: 'bounce', label: 'Bounce' },
                { key: 'shake', label: 'Impact Shake' },
                { key: 'wave', label: 'Wave sway' },
                { key: 'spin', label: 'Spin' },
            ],
        },
        {
            label: 'Entrance / Motion',
            effects: [
                { key: 'slide-up', label: 'Slide Up' },
                { key: 'slide-left', label: 'Slide from Left' },
                { key: 'zoom-in', label: 'Elastic Zoom' },
            ]
        },
        {
            label: 'Camera',
            effects: [
                { key: 'parallax', label: '3D Parallax Pan' },
            ]
        },
        {
            label: 'Lighting / Color',
            effects: [
                { key: 'glow', label: 'Glow pulse' },
                { key: 'rim', label: 'Rim light' },
                { key: 'shine', label: 'Shine sweep' },
                { key: 'gold_pulse', label: 'Gold pulse' },
                { key: 'flicker', label: 'Flicker' },
                { key: 'fade', label: 'Fade pulse' },
                { key: 'rainbow', label: 'Rainbow Cycle' },
            ],
        },
        {
            label: 'Particle',
            effects: [
                { key: 'sparkle', label: 'Sparkle' },
                { key: 'color_shift', label: 'Color shift' },
            ],
        },
    ];

    // -----------------------------------------------------------------------
    // Region state
    // -----------------------------------------------------------------------
    const regions = (JSON.parse(dataNode.textContent) || []).map(function (item, index) {
        return {
            key: item.key || ('region-' + index),
            label: item.label || ('Region ' + (index + 1)),
            source: item.source || 'manual',
            x: Number(item.x) || 0,
            y: Number(item.y) || 0,
            width: Number(item.width) || MIN,
            height: Number(item.height) || MIN,
            effects: Array.isArray(item.effects) ? item.effects.slice() : [],
        };
    });

    let selected = regions.length ? 0 : -1;
    let drag = null;

    // -----------------------------------------------------------------------
    // Canvas helpers
    // -----------------------------------------------------------------------
    function colorFor(source, isSelected) {
        if (isSelected) return '#3fb950';
        if (source === 'ocr' || source === 'title') return '#f0b429';
        if (source === 'card') return '#3ecfcf';
        if (source === 'button') return '#f778ba';
        if (source === 'prop') return '#ff9f43';
        if (source === 'yolo' || source === 'sam') return '#4f8cff';
        return '#bc8cff';
    }

    function toBox(item, width, height) {
        return {
            x: item.x * width,
            y: item.y * height,
            w: item.width * width,
            h: item.height * height,
        };
    }

    function handlesFor(box) {
        const x1 = box.x, y1 = box.y, x2 = box.x + box.w, y2 = box.y + box.h;
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        return [
            { id: 'nw', x: x1, y: y1 }, { id: 'n', x: mx, y: y1 }, { id: 'ne', x: x2, y: y1 },
            { id: 'e', x: x2, y: my }, { id: 'se', x: x2, y: y2 }, { id: 's', x: mx, y: y2 },
            { id: 'sw', x: x1, y: y2 }, { id: 'w', x: x1, y: my },
        ];
    }

    function hitHandle(px, py, box) {
        const half = HANDLE + 2;
        const hits = handlesFor(box);
        for (let i = 0; i < hits.length; i++) {
            if (Math.abs(px - hits[i].x) <= half && Math.abs(py - hits[i].y) <= half) {
                return hits[i].id;
            }
        }
        return null;
    }

    function clampRegion(item) {
        item.x = Math.min(Math.max(item.x, 0), 1 - MIN);
        item.y = Math.min(Math.max(item.y, 0), 1 - MIN);
        item.width = Math.min(Math.max(item.width, MIN), 1 - item.x);
        item.height = Math.min(Math.max(item.height, MIN), 1 - item.y);
    }

    function applyResize(item, handle, nx, ny, start) {
        let x1 = start.x, y1 = start.y, x2 = start.x + start.width, y2 = start.y + start.height;
        if (handle.indexOf('n') !== -1) y1 = Math.min(ny, y2 - MIN);
        if (handle.indexOf('s') !== -1) y2 = Math.max(ny, y1 + MIN);
        if (handle.indexOf('w') !== -1) x1 = Math.min(nx, x2 - MIN);
        if (handle.indexOf('e') !== -1) x2 = Math.max(nx, x1 + MIN);
        item.x = Math.min(Math.max(x1, 0), 1 - MIN);
        item.y = Math.min(Math.max(y1, 0), 1 - MIN);
        item.width = Math.min(Math.max(x2 - item.x, MIN), 1 - item.x);
        item.height = Math.min(Math.max(y2 - item.y, MIN), 1 - item.y);
    }

    function draw() {
        const rect = canvas.getBoundingClientRect();
        const width = rect.width, height = rect.height;
        if (!width || !height) return;
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
        const ctx = canvas.getContext('2d');
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, width, height);

        regions.forEach(function (item, index) {
            const box = toBox(item, width, height);
            const isSelected = index === selected;
            const color = colorFor(item.source, isSelected);

            ctx.save();
            if (isSelected) {
                ctx.fillStyle = 'rgba(63, 185, 80, 0.16)';
                ctx.fillRect(box.x, box.y, box.w, box.h);
            }
            ctx.strokeStyle = color;
            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.strokeRect(box.x, box.y, box.w, box.h);

            const text = item.label || item.source;
            ctx.font = '12px sans-serif';
            const textWidth = ctx.measureText(text).width;
            ctx.fillStyle = color;
            ctx.fillRect(box.x, Math.max(0, box.y - 18), textWidth + 10, 18);
            ctx.fillStyle = '#0e1116';
            ctx.fillText(text, box.x + 5, Math.max(12, box.y - 5));

            if (isSelected) {
                ctx.fillStyle = '#fff';
                ctx.strokeStyle = color;
                ctx.lineWidth = 1;
                handlesFor(box).forEach(function (h) {
                    ctx.fillRect(h.x - HANDLE / 2, h.y - HANDLE / 2, HANDLE, HANDLE);
                    ctx.strokeRect(h.x - HANDLE / 2, h.y - HANDLE / 2, HANDLE, HANDLE);
                });
            }
            ctx.restore();
        });
    }

    // -----------------------------------------------------------------------
    // Effect panel — rendered per selected region
    // -----------------------------------------------------------------------
    function renderEffectPanel() {
        if (!effectPanel) return;
        if (selected < 0 || !regions[selected]) {
            effectPanel.innerHTML = '<p class="hint">Select a region above to choose its animation effects.</p>';
            return;
        }
        const item = regions[selected];
        const current = new Set(item.effects || []);

        let html = '<h3 class="effect-panel-title">'
            + 'Effects for <span class="tag tag-' + item.source + '">'
            + item.source + '</span> '
            + escapeHtml(item.label) + '</h3>';

        const bgSize = (100 / item.width).toFixed(2) + '% ' + (100 / item.height).toFixed(2) + '%';
        const bgPosX = item.width < 1 ? (item.x / (1 - item.width) * 100).toFixed(2) + '%' : '0%';
        const bgPosY = item.height < 1 ? (item.y / (1 - item.height) * 100).toFixed(2) + '%' : '0%';
        const bgStyle = 'background-image: url(' + image.src + '); ' +
                        'background-size: ' + bgSize + '; ' +
                        'background-position: ' + bgPosX + ' ' + bgPosY + ';';

        EFFECT_GROUPS.forEach(function (group) {
            html += '<div class="effect-group"><div class="effect-group-label">' + group.label + '</div>';
            html += '<div class="effect-grid">';
            group.effects.forEach(function (effect) {
                const checked = current.has(effect.key) ? ' checked' : '';
                html += '<label class="effect-option" title="Preview: ' + effect.label + '">'
                     +  '<input type="checkbox" class="region-effect-check"'
                     +  ' data-key="' + effect.key + '"' + checked + '>'
                     +  '<div class="effect-preview effect-preview-' + effect.key + '" style="' + bgStyle + '"></div>'
                     +  '<span style="font-weight:500;">' + effect.label + '</span>'
                     +  '</label>';
            });
            html += '</div></div>';
        });

        effectPanel.innerHTML = html;

        // Wire up checkboxes
        effectPanel.querySelectorAll('.region-effect-check').forEach(function (cb) {
            cb.addEventListener('change', function () {
                const key = cb.getAttribute('data-key');
                const effects = regions[selected].effects;
                if (cb.checked) {
                    if (!effects.includes(key)) effects.push(key);
                } else {
                    const idx = effects.indexOf(key);
                    if (idx !== -1) effects.splice(idx, 1);
                }
                syncHidden();
            });
        });
    }

    function escapeHtml(str) {
        return String(str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // -----------------------------------------------------------------------
    // Region list sidebar
    // -----------------------------------------------------------------------
    function renderList() {
        if (!list) return;
        list.innerHTML = '';
        regions.forEach(function (item, index) {
            const row = document.createElement('li');
            row.className = index === selected ? 'is-selected' : '';
            const effectCount = (item.effects || []).length;
            const badge = effectCount > 0
                ? '<span class="effect-badge">' + effectCount + ' fx</span>'
                : '';
            row.innerHTML =
                '<button type="button" class="region-pick" data-index="' + index + '" style="display:flex;align-items:center;gap:6px;">'
                + '<span class="tag tag-' + item.source + '">' + item.source + '</span> '
                + escapeHtml(item.label)
                + badge
                + '</button>'
                + '<button type="button" class="region-remove" data-index="' + index + '" aria-label="Remove">Remove</button>';
            list.appendChild(row);
        });
    }

    // -----------------------------------------------------------------------
    // Hidden input sync
    // -----------------------------------------------------------------------
    function syncHidden() {
        if (regionsInput) {
            regionsInput.value = JSON.stringify(regions);
        }
        // Expose on window so dialog.js can read the live array
        // without depending on the hidden input being serialized first.
        window.__adjustRegions = regions;
    }

    function sync() {
        regions.forEach(clampRegion);
        syncHidden();
        renderList();
        renderEffectPanel();
        draw();
    }

    // -----------------------------------------------------------------------
    // Pointer helpers
    // -----------------------------------------------------------------------
    function pointerOnImage(event) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top,
            nx: (event.clientX - rect.left) / rect.width,
            ny: (event.clientY - rect.top) / rect.height,
            width: rect.width,
            height: rect.height,
        };
    }

    function hitRegion(px, py, width, height) {
        for (let i = regions.length - 1; i >= 0; i--) {
            const box = toBox(regions[i], width, height);
            if (px >= box.x && py >= box.y && px <= box.x + box.w && py <= box.y + box.h) {
                return i;
            }
        }
        return -1;
    }

    function setCursor(px, py, width, height) {
        if (!wrap) return;
        if (selected >= 0) {
            const handle = hitHandle(px, py, toBox(regions[selected], width, height));
            if (handle) { canvas.style.cursor = cursors[handle]; return; }
        }
        const index = hitRegion(px, py, width, height);
        canvas.style.cursor = index >= 0 ? 'move' : 'default';
    }

    // -----------------------------------------------------------------------
    // Mouse events
    // -----------------------------------------------------------------------
    canvas.addEventListener('mousedown', function (event) {
        if (event.button !== 0) return;
        event.preventDefault();
        const point = pointerOnImage(event);

        if (selected >= 0) {
            const handle = hitHandle(point.x, point.y, toBox(regions[selected], point.width, point.height));
            if (handle) {
                const item = regions[selected];
                drag = {
                    kind: 'resize', handle: handle, index: selected,
                    start: { x: item.x, y: item.y, width: item.width, height: item.height }
                };
                return;
            }
        }

        const index = hitRegion(point.x, point.y, point.width, point.height);
        selected = index;
        if (index >= 0) {
            const item = regions[index];
            drag = {
                kind: 'move', index: index,
                offsetX: point.nx - item.x, offsetY: point.ny - item.y
            };
        }
        sync();
    });

    canvas.addEventListener('mousemove', function (event) {
        const point = pointerOnImage(event);
        if (!drag) { setCursor(point.x, point.y, point.width, point.height); return; }
        const item = regions[drag.index];
        if (drag.kind === 'move') {
            item.x = point.nx - drag.offsetX;
            item.y = point.ny - drag.offsetY;
            clampRegion(item);
        } else {
            applyResize(item, drag.handle, point.nx, point.ny, drag.start);
        }
        draw();
    });

    window.addEventListener('mouseup', function () {
        if (drag) { drag = null; sync(); }
    });

    // -----------------------------------------------------------------------
    // Region list click events
    // -----------------------------------------------------------------------
    if (list) {
        list.addEventListener('click', function (event) {
            const pick = event.target.closest('.region-pick');
            const remove = event.target.closest('.region-remove');
            if (remove) {
                const index = Number(remove.getAttribute('data-index'));
                if (regions.length <= 1) {
                    if (typeof window.showAppDialog === 'function') {
                        window.showAppDialog('Cannot remove the last region',
                            'Keep at least one region. The GIF needs a box to animate.');
                    }
                    return;
                }
                regions.splice(index, 1);
                selected = Math.min(selected, regions.length - 1);
                sync();
                return;
            }
            if (pick) {
                selected = Number(pick.getAttribute('data-index'));
                sync();
            }
        });
    }

    // -----------------------------------------------------------------------
    // Keyboard shortcuts
    // -----------------------------------------------------------------------
    window.addEventListener('keydown', function (event) {
        if (event.target && (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA')) return;
        if (selected < 0 || !regions[selected]) return;
        const item = regions[selected];
        const step = event.shiftKey ? 0.01 : 0.004;
        if (event.key === 'ArrowLeft') item.x -= step;
        else if (event.key === 'ArrowRight') item.x += step;
        else if (event.key === 'ArrowUp') item.y -= step;
        else if (event.key === 'ArrowDown') item.y += step;
        else if (event.key === 'Delete' || event.key === 'Backspace') {
            if (regions.length > 1) {
                regions.splice(selected, 1);
                selected = Math.min(selected, regions.length - 1);
            } else if (typeof window.showAppDialog === 'function') {
                window.showAppDialog('Cannot remove the last region',
                    'Keep at least one region. The GIF needs a box to animate.');
            }
        } else { return; }
        event.preventDefault();
        sync();
    });

    // -----------------------------------------------------------------------
    // Form submit
    // -----------------------------------------------------------------------
    if (form) {
        form.addEventListener('submit', sync, true);
    }

    // -----------------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------------
    if (image.complete && image.naturalWidth) { sync(); }
    image.addEventListener('load', sync);
    window.addEventListener('resize', draw);
    sync();
})();
