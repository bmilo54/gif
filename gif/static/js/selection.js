(function () {
    const image = document.getElementById('source-image');
    const canvas = document.getElementById('overlay');
    const wrap = document.getElementById('canvas-wrap');
    const dataNode = document.getElementById('detections-data');
    const countNode = document.getElementById('selected-count');
    const idsInput = document.getElementById('detection-ids');
    const regionsInput = document.getElementById('manual-regions');
    const form = document.getElementById('animate-form');
    const hint = document.getElementById('mode-hint');
    const submit = document.getElementById('animate-submit');

    if (!image || !canvas || !dataNode) {
        return;
    }

    const detections = JSON.parse(dataNode.textContent) || [];
    const selected = new Set();
    const manualRegions = [];
    let mode = 'select';
    let drag = null;

    const hints = {
        select: 'Click a yellow or blue box to select it. Use Draw region for the character YOLO missed.',
        draw: 'Drag on the image to box the character or any region detection missed. Click a drawn box to remove it.',
    };

    function colorFor(source, isSelected) {
        if (isSelected) {
            return '#3fb950';
        }
        if (source === 'ocr') {
            return '#f0b429';
        }
        if (source === 'manual') {
            return '#bc8cff';
        }
        return '#4f8cff';
    }

    function toBox(item, width, height) {
        return {
            x: item.x * width,
            y: item.y * height,
            w: item.width * width,
            h: item.height * height,
        };
    }

    function drawBox(ctx, box, source, label, isSelected) {
        const color = colorFor(source, isSelected);
        ctx.save();
        if (isSelected) {
            ctx.fillStyle = 'rgba(63, 185, 80, 0.18)';
            ctx.fillRect(box.x, box.y, box.w, box.h);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        const text = label || source;
        ctx.font = '12px sans-serif';
        const textWidth = ctx.measureText(text).width;
        ctx.fillStyle = color;
        ctx.fillRect(box.x, Math.max(0, box.y - 18), textWidth + 10, 18);
        ctx.fillStyle = '#0e1116';
        ctx.fillText(text, box.x + 5, Math.max(12, box.y - 5));
        ctx.restore();
    }

    function draw() {
        const width = image.clientWidth;
        const height = image.clientHeight;
        const ratio = window.devicePixelRatio || 1;

        canvas.width = width * ratio;
        canvas.height = height * ratio;

        const ctx = canvas.getContext('2d');
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, width, height);

        detections.forEach(function (item) {
            const label = item.label + ' ' + Math.round(item.confidence * 100) + '%';
            drawBox(ctx, toBox(item, width, height), item.source, label, selected.has(item.id));
        });

        manualRegions.forEach(function (item, index) {
            drawBox(
                ctx,
                toBox(item, width, height),
                'manual',
                'Manual ' + (index + 1),
                true
            );
        });

        if (drag) {
            const x = Math.min(drag.startX, drag.currentX);
            const y = Math.min(drag.startY, drag.currentY);
            const w = Math.abs(drag.currentX - drag.startX);
            const h = Math.abs(drag.currentY - drag.startY);
            ctx.strokeStyle = '#bc8cff';
            ctx.setLineDash([6, 4]);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);
        }
    }

    function syncForm() {
        idsInput.value = Array.from(selected).join(',');
        regionsInput.value = JSON.stringify(manualRegions);
        const total = selected.size + manualRegions.length;
        if (countNode) {
            countNode.textContent = String(total);
        }
        if (submit) {
            submit.disabled = total === 0;
        }

        document.querySelectorAll('#detection-table tr[data-id]').forEach(function (row) {
            const id = Number(row.getAttribute('data-id'));
            const on = selected.has(id);
            row.classList.toggle('is-selected', on);
            const check = row.querySelector('.row-check');
            if (check) {
                check.checked = on;
            }
        });
    }

    function toggleDetection(id) {
        if (selected.has(id)) {
            selected.delete(id);
        } else {
            selected.add(id);
        }
        syncForm();
        draw();
    }

    function pointerOnImage(event) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top,
            width: rect.width,
            height: rect.height,
        };
    }

    function hitDetection(px, py, width, height) {
        const hits = detections.filter(function (item) {
            const box = toBox(item, width, height);
            return px >= box.x && py >= box.y && px <= box.x + box.w && py <= box.y + box.h;
        });
        hits.sort(function (a, b) {
            return (a.width * a.height) - (b.width * b.height);
        });
        return hits[0] || null;
    }

    function hitManualIndex(px, py, width, height) {
        for (let i = manualRegions.length - 1; i >= 0; i -= 1) {
            const box = toBox(manualRegions[i], width, height);
            if (px >= box.x && py >= box.y && px <= box.x + box.w && py <= box.y + box.h) {
                return i;
            }
        }
        return -1;
    }

    canvas.addEventListener('mousedown', function (event) {
        if (event.button !== 0) {
            return;
        }
        const point = pointerOnImage(event);

        if (mode === 'draw') {
            const manualIndex = hitManualIndex(point.x, point.y, point.width, point.height);
            if (manualIndex >= 0 && !event.shiftKey) {
                manualRegions.splice(manualIndex, 1);
                syncForm();
                draw();
                return;
            }
            drag = {
                startX: point.x,
                startY: point.y,
                currentX: point.x,
                currentY: point.y,
            };
            return;
        }

        const hit = hitDetection(point.x, point.y, point.width, point.height);
        if (hit) {
            toggleDetection(hit.id);
        }
    });

    canvas.addEventListener('mousemove', function (event) {
        if (!drag) {
            return;
        }
        const point = pointerOnImage(event);
        drag.currentX = point.x;
        drag.currentY = point.y;
        draw();
    });

    function finishDrag(event) {
        if (!drag) {
            return;
        }
        const point = pointerOnImage(event);
        const x1 = Math.min(drag.startX, point.x) / point.width;
        const y1 = Math.min(drag.startY, point.y) / point.height;
        const x2 = Math.max(drag.startX, point.x) / point.width;
        const y2 = Math.max(drag.startY, point.y) / point.height;
        drag = null;

        const width = x2 - x1;
        const height = y2 - y1;
        if (width >= 0.02 && height >= 0.02) {
            manualRegions.push({ x: x1, y: y1, width: width, height: height });
            syncForm();
        }
        draw();
    }

    window.addEventListener('mouseup', finishDrag);

    document.querySelectorAll('[data-mode]').forEach(function (button) {
        button.addEventListener('click', function () {
            mode = button.getAttribute('data-mode');
            document.querySelectorAll('[data-mode]').forEach(function (other) {
                other.classList.toggle('is-active', other === button);
            });
            if (wrap) {
                wrap.classList.toggle('is-draw', mode === 'draw');
            }
            if (hint) {
                hint.textContent = hints[mode];
            }
        });
    });

    document.querySelectorAll('.row-check').forEach(function (check) {
        check.addEventListener('change', function () {
            toggleDetection(Number(check.getAttribute('data-id')));
        });
    });

    document.querySelectorAll('#detection-table tr[data-id]').forEach(function (row) {
        row.addEventListener('click', function (event) {
            if (event.target.classList.contains('row-check')) {
                return;
            }
            toggleDetection(Number(row.getAttribute('data-id')));
        });
    });

    if (form) {
        form.addEventListener('submit', function (event) {
            syncForm();
            if (selected.size + manualRegions.length === 0) {
                event.preventDefault();
            }
        });
    }

    if (image.complete) {
        draw();
    } else {
        image.addEventListener('load', draw);
    }
    window.addEventListener('resize', draw);
    syncForm();
})();
draw();
    }
image.addEventListener('load', draw);
window.addEventListener('resize', draw);
syncForm();
}) ();
