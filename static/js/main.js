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
        const ext = f.name.split('.').pop().toLowerCase();
        const imgExts = ['png', 'jpg', 'jpeg', 'bmp', 'webp'];
        const ext3d = ['obj', 'npy', 'npz', 'bin', 'ply', 'stl', 'glb', 'fbx'];
        const isImg = f.type.match(/image\/.*/) || imgExts.includes(ext);
        const is3d = ext3d.includes(ext);
        if (!isImg && !is3d) { showToast('Please select a valid image or 3D file.', 'error'); return; }

        coverFile = f;
        document.getElementById('cover-filename').textContent = f.name;

        if (isImg) {
            const reader = new FileReader();
            reader.onload = e => document.getElementById('cover-preview').src = e.target.result;
            reader.readAsDataURL(f);
        } else {
            // 3D file — show icon instead of image preview
            document.getElementById('cover-preview').src =
                'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="%236366f1" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>';
        }
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

    // 3D file zone — dynamic input to bypass Windows 'Image files' filter caching
    const _zone3d = document.getElementById('secret-3d-drop-zone');
    if (_zone3d) {
        function _pick3dFile() {
            const tmp = document.createElement('input');
            tmp.type = 'file'; // NO accept → shows ALL file types
            tmp.onchange = e => {
                const f = e.target.files[0];
                if (!f) return;
                window._secret3dFile = f;
                document.getElementById('secret-3d-filename').textContent = '✅ ' + f.name;
                _zone3d.style.borderColor = 'var(--accent)';
            };
            tmp.click();
        }
        _zone3d.addEventListener('click', _pick3dFile);
        _zone3d.addEventListener('dragover', e => { e.preventDefault(); _zone3d.classList.add('dragover'); });
        _zone3d.addEventListener('dragleave', () => _zone3d.classList.remove('dragover'));
        _zone3d.addEventListener('drop', e => {
            e.preventDefault(); _zone3d.classList.remove('dragover');
            const f = e.dataTransfer.files[0];
            if (!f) return;
            window._secret3dFile = f;
            document.getElementById('secret-3d-filename').textContent = '✅ ' + f.name;
            _zone3d.style.borderColor = 'var(--accent)';
        });
    }

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
            const f = window._secret3dFile || document.getElementById('secret-3d-input')?.files[0];
            if (!f) { showToast('Please select or drop a 3D data file.', 'error'); return; }
            fd.append('secret_3d', f);
        }

        setLoading('btn-hide', true);
        try {
            const resp = await fetch('/hide/process', { method: 'POST', body: fd });
            const data = await resp.json();
            if (data.error) { showToast(data.error, 'error'); return; }

            // Display result
            document.getElementById('result-stego-img').src = data.stego_image || '';
            document.getElementById('metric-psnr').textContent = data.psnr + (data.cover_type === 'image' ? ' dB' : '');
            document.getElementById('metric-ssim').textContent = data.ssim;
            document.getElementById('metric-robust').textContent = data.robustness + (data.cover_type === 'image' ? '%' : '');
            document.getElementById('model-badge-text').textContent = data.model_used;

            // Show correct download button
            const dlImg = document.getElementById('download-stego');
            const dl3d = document.getElementById('download-stego-3d');

            if (data.cover_type === 'image') {
                dlImg.href = data.stego_image;
                dlImg.download = data.stego_filename || 'stego.png';
                dlImg.style.display = 'inline-flex';
                dl3d.style.display = 'none';
            } else {
                // 3D stego
                window._stego3dData = data.stego_3d_data;
                window._stego3dFilename = data.stego_filename || ('stego.' + data.stego_ext);
                dlImg.style.display = 'none';
                dl3d.style.display = 'inline-flex';
                // Show a 3D file icon in the preview slot
                document.getElementById('result-stego-img').style.display = 'none';
            }
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

// Download stego 3D file
function downloadStego3D() {
    const b64 = window._stego3dData;
    const fname = window._stego3dFilename || 'stego_3d.bin';
    if (!b64) { showToast('No stego 3D data available.', 'error'); return; }
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const blob = new Blob([bytes], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = fname; a.click();
    URL.revokeObjectURL(url);
    showToast('3D stego file downloaded!', 'info');
}
