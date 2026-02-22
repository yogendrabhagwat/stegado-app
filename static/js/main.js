/**
 * main.js – Stegado Flask App
 * Handles: Hide page (encode), Extract page (decode), flash auto-dismiss,
 * password strength meter, drag-and-drop, segment controls.
 */

// ─── Flash Auto-Dismiss ───────────────────────────────────────────────────────
document.querySelectorAll('.flash-msg').forEach(el => {
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transform = 'translateX(20px)';
        el.style.transition = 'all 0.4s ease';
        setTimeout(() => el.remove(), 400);
    }, 4000);
});

// ─── Password Strength Meter ──────────────────────────────────────────────────
function initPasswordStrength(inputId, barId, labelId) {
    const inp = document.getElementById(inputId);
    const bar = document.getElementById(barId);
    const lbl = document.getElementById(labelId);
    if (!inp || !bar) return;

    const levels = [
        { label: 'Very Weak', color: '#ef4444', pct: 10 },
        { label: 'Weak', color: '#f97316', pct: 30 },
        { label: 'Fair', color: '#eab308', pct: 55 },
        { label: 'Strong', color: '#22c55e', pct: 80 },
        { label: 'Very Strong', color: '#4ade80', pct: 100 },
    ];

    inp.addEventListener('input', () => {
        const v = inp.value;
        let score = 0;
        if (v.length >= 8) score++;
        if (v.length >= 12) score++;
        if (/[A-Z]/.test(v)) score++;
        if (/[0-9]/.test(v)) score++;
        if (/[^A-Za-z0-9]/.test(v)) score++;

        const lvl = levels[Math.min(score, 4)];
        bar.style.width = lvl.pct + '%';
        bar.style.background = lvl.color;
        if (lbl) { lbl.textContent = v ? lvl.label : ''; lbl.style.color = lvl.color; }
    });
}
initPasswordStrength('password', 'strength-bar', 'strength-label');
initPasswordStrength('aes-password', 'aes-strength-bar', 'aes-strength-label');

// ─── Drag & Drop Helper ───────────────────────────────────────────────────────
function setupDragDrop(zoneId, inputId, onFile) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) onFile(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', e => { if (e.target.files.length) onFile(e.target.files[0]); });
}

// ─── HIDE PAGE ────────────────────────────────────────────────────────────────
(function initHidePage() {
    if (!document.getElementById('hide-form')) return;

    let coverFile = null;

    // Cover drop
    setupDragDrop('cover-drop-zone', 'cover-file-input', f => {
        if (!f.type.match(/image\/.*/)) { showToast('Please select a valid image.', 'error'); return; }
        coverFile = f;
        document.getElementById('cover-filename').textContent = f.name;
        const reader = new FileReader();
        reader.onload = e => document.getElementById('cover-preview').src = e.target.result;
        reader.readAsDataURL(f);
        show('cover-step-2');
        hide('cover-drop-zone');
        hide('hide-result');
    });

    document.getElementById('clear-cover')?.addEventListener('click', () => {
        coverFile = null;
        document.getElementById('cover-file-input').value = '';
        show('cover-drop-zone');
        hide('cover-step-2');
        hide('hide-result');
    });

    // Secret type switching
    document.querySelectorAll('input[name="secret_type"]').forEach(r => {
        r.addEventListener('change', () => {
            const t = r.value;
            toggleEl('secret-text-area', t === 'text');
            toggleEl('secret-image-area', t === 'image');
            toggleEl('secret-3d-area', t === '3d');
        });
    });

    // 3D file drag-drop zone
    setupDragDrop('secret-3d-drop-zone', 'secret-3d-input', f => {
        document.getElementById('secret-3d-filename').textContent = '✅ ' + f.name;
        document.getElementById('secret-3d-drop-zone').style.borderColor = 'var(--accent)';
    });

    // Submit
    document.getElementById('btn-hide')?.addEventListener('click', async () => {
        if (!coverFile) { showToast('Please upload a cover image.', 'error'); return; }
        const password = document.getElementById('aes-password')?.value.trim();
        if (!password || password.length < 4) { showToast('Password must be ≥ 4 characters.', 'error'); return; }

        const secretType = document.querySelector('input[name="secret_type"]:checked')?.value || 'text';
        const coverMode = document.querySelector('input[name="cover_mode"]:checked')?.value || 'Universal';

        const fd = new FormData();
        fd.append('cover_image', coverFile);
        fd.append('password', password);
        fd.append('secret_type', secretType);
        fd.append('cover_mode', coverMode);

        if (secretType === 'text') {
            const txt = document.getElementById('secret-text')?.value.trim();
            if (!txt) { showToast('Please enter a secret message.', 'error'); return; }
            fd.append('secret_text', txt);
        } else if (secretType === 'image') {
            const f = document.getElementById('secret-image-input')?.files[0];
            if (!f) { showToast('Please select a secret image.', 'error'); return; }
            fd.append('secret_image', f);
        } else if (secretType === '3d') {
            const f = document.getElementById('secret-3d-input')?.files[0];
            if (!f) { showToast('Please select a 3D data file.', 'error'); return; }
            fd.append('secret_3d', f);
        }

        setLoading('btn-hide', true);
        try {
            const resp = await fetch('/hide/process', { method: 'POST', body: fd });
            const data = await resp.json();
            if (data.error) { showToast(data.error, 'error'); return; }

            // Display result
            document.getElementById('result-stego-img').src = data.stego_image;
            document.getElementById('metric-psnr').textContent = data.psnr + ' dB';
            document.getElementById('metric-ssim').textContent = data.ssim;
            document.getElementById('metric-robust').textContent = data.robustness + '%';
            document.getElementById('model-badge-text').textContent = data.model_used;

            // Download link
            const dl = document.getElementById('download-stego');
            dl.href = data.stego_image;
            dl.download = 'stego_' + coverFile.name.split('.')[0] + '.png';

            show('hide-result');
            document.getElementById('hide-result').scrollIntoView({ behavior: 'smooth' });
        } catch (e) {
            showToast('Network error: ' + e.message, 'error');
        } finally {
            setLoading('btn-hide', false);
        }
    });
})();

// ─── EXTRACT PAGE ─────────────────────────────────────────────────────────────
(function initExtractPage() {
    if (!document.getElementById('extract-form')) return;

    let stegoFile = null;

    setupDragDrop('stego-drop-zone', 'stego-file-input', f => {
        stegoFile = f;
        document.getElementById('stego-filename').textContent = f.name;
        const reader = new FileReader();
        reader.onload = e => document.getElementById('stego-preview').src = e.target.result;
        reader.readAsDataURL(f);
        show('stego-step-2');
        hide('stego-drop-zone');
        hide('extract-result');
    });

    document.getElementById('clear-stego')?.addEventListener('click', () => {
        stegoFile = null;
        document.getElementById('stego-file-input').value = '';
        show('stego-drop-zone');
        hide('stego-step-2');
        hide('extract-result');
    });

    document.getElementById('btn-extract')?.addEventListener('click', async () => {
        if (!stegoFile) { showToast('Please upload a stego image.', 'error'); return; }
        const password = document.getElementById('extract-password')?.value.trim();
        if (!password || password.length < 4) { showToast('Password must be ≥ 4 characters.', 'error'); return; }

        const fd = new FormData();
        fd.append('stego_image', stegoFile);
        fd.append('password', password);

        setLoading('btn-extract', true);
        try {
            const resp = await fetch('/extract/process', { method: 'POST', body: fd });
            const data = await resp.json();
            if (data.error) { showToast(data.error, 'error'); return; }

            const result = data.result;
            const resBox = document.getElementById('extract-result');
            const textOut = document.getElementById('decoded-message');
            const imgOut = document.getElementById('decoded-image');

            hide('result-type-text');
            hide('result-type-image');
            hide('result-type-3d');

            if (result.type === 'text') {
                textOut.textContent = result.content;
                show('result-type-text');
            } else if (result.type === 'image') {
                imgOut.src = result.content;
                show('result-type-image');
            } else {
                document.getElementById('decoded-3d-info').textContent = result.content;
                // Store data for download button
                window._3dData = { data: result.data, ext: result.ext };
                show('result-type-3d');
            }
            show('extract-result');
            resBox.scrollIntoView({ behavior: 'smooth' });
        } catch (e) {
            showToast('Network error: ' + e.message, 'error');
        } finally {
            setLoading('btn-extract', false);
        }
    });
})();

// ─── Helpers ──────────────────────────────────────────────────────────────────
function show(id) { document.getElementById(id)?.classList.remove('hidden'); }
function hide(id) { document.getElementById(id)?.classList.add('hidden'); }
function toggleEl(id, vis) { vis ? show(id) : hide(id); }

function setLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    if (loading) {
        btn._original = btn.innerHTML;
        btn.innerHTML = '<span class="spinner"></span><span>Processing…</span>';
        btn.disabled = true;
    } else {
        btn.innerHTML = btn._original || btn.innerHTML;
        btn.disabled = false;
    }
}

function showToast(msg, type = 'info') {
    let container = document.querySelector('.flash-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'flash-container';
        document.body.appendChild(container);
    }
    const el = document.createElement('div');
    el.className = `flash-msg ${type}`;
    el.textContent = msg;
    container.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0'; el.style.transform = 'translateX(20px)';
        el.style.transition = 'all 0.4s ease';
        setTimeout(() => el.remove(), 400);
    }, 4000);
}

// Share button
document.getElementById('btn-share')?.addEventListener('click', () => {
    const img = document.getElementById('result-stego-img')?.src;
    if (!img) return;
    if (navigator.share) {
        navigator.share({ title: 'Stegado – Stego Image', text: 'Secret hidden inside!' })
            .catch(() => { });
    } else {
        navigator.clipboard.writeText(window.location.href);
        showToast('Link copied to clipboard!', 'info');
    }
});
