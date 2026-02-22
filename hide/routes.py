import io
import base64
import os
from flask import render_template, request, jsonify
from flask_login import login_required, current_user
from hide import hide_bp
from extensions import db
from models import History
from utils.image_utils import (
    embed_secret, compute_robustness_score,
    secret_text_to_bytes, secret_image_to_bytes, secret_3d_to_bytes
)
from PIL import Image

ALLOWED_COVER = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
ALLOWED_SECRET_IMG = {'png', 'jpg', 'jpeg', 'bmp'}


def _allowed(filename, exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts


@hide_bp.route('/')
@login_required
def hide_page():
    return render_template('hide/hide.html')


@hide_bp.route('/process', methods=['POST'])
@login_required
def process():
    try:
        # ── Inputs ────────────────────────────────────────────────────────────
        cover_file = request.files.get('cover_image')
        password = request.form.get('password', '').strip()
        secret_type = request.form.get('secret_type', 'text')   # text / image / 3d
        cover_mode = request.form.get('cover_mode', 'Universal') # Universal / 2D / 3D

        if not cover_file or not cover_file.filename:
            return jsonify({'error': 'No cover image provided.'}), 400
        if not _allowed(cover_file.filename, ALLOWED_COVER):
            return jsonify({'error': 'Unsupported cover image format.'}), 400
        if len(password) < 4:
            return jsonify({'error': 'Password must be at least 4 characters.'}), 400

        # ── Load cover image ──────────────────────────────────────────────────
        cover_img = Image.open(cover_file.stream)

        # ── Prepare secret bytes ──────────────────────────────────────────────
        if secret_type == 'text':
            text = request.form.get('secret_text', '').strip()
            if not text:
                return jsonify({'error': 'Secret text is empty.'}), 400
            secret_bytes = secret_text_to_bytes(text)

        elif secret_type == 'image':
            secret_file = request.files.get('secret_image')
            if not secret_file or not _allowed(secret_file.filename, ALLOWED_SECRET_IMG):
                return jsonify({'error': 'Please upload a valid secret image.'}), 400
            secret_bytes = secret_image_to_bytes(secret_file.stream)

        elif secret_type == '3d':
            secret_file = request.files.get('secret_3d')
            if not secret_file:
                return jsonify({'error': 'Please upload a 3D data file.'}), 400
            secret_bytes = secret_3d_to_bytes(secret_file.stream)

        else:
            return jsonify({'error': 'Unknown secret type.'}), 400

        # ── Embed ─────────────────────────────────────────────────────────────
        stego_img, psnr, ssim = embed_secret(cover_img, secret_bytes, password)
        robustness = compute_robustness_score(psnr, ssim)
        try:
            from core.model_loader import get_active_model
            model_label = get_active_model(cover_mode)
        except Exception:
            model_label = f'{cover_mode} (Signal-Processing Engine)'

        # ── Encode stego as base64 PNG ────────────────────────────────────────
        buf = io.BytesIO()
        stego_img.save(buf, format='PNG', optimize=False)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        stego_data_url = f'data:image/png;base64,{b64}'

        # ── Save History ──────────────────────────────────────────────────────
        entry = History(
            user_id=current_user.id,
            operation='hide',
            file_name=cover_file.filename,
            model_used=model_label,
            psnr_value=psnr,
            robustness_score=robustness,
            result_type=secret_type,
        )
        db.session.add(entry)
        db.session.commit()

        return jsonify({
            'success': True,
            'stego_image': stego_data_url,
            'psnr': psnr,
            'ssim': ssim,
            'robustness': robustness,
            'model_used': model_label,
            'secret_type': secret_type,
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
