import io
import base64
import os
from flask import render_template, request, jsonify
from flask_login import login_required, current_user
from hide import hide_bp
from extensions import db
from models import History
from utils.neural_stego import (
    neural_embed, compute_robustness_score,
    secret_text_to_bytes, secret_image_to_bytes, secret_3d_to_bytes
)
from utils.cover_3d import embed_in_3d, is_3d_file
from PIL import Image

ALLOWED_COVER_IMG = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
ALLOWED_COVER_3D  = {'obj', 'npy', 'npz', 'bin', 'ply', 'stl', 'glb', 'fbx'}
ALLOWED_SECRET_IMG = {'png', 'jpg', 'jpeg', 'bmp'}


def _ext(filename: str) -> str:
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

def _allowed(filename, exts):
    return _ext(filename) in exts


@hide_bp.route('/')
@login_required
def hide_page():
    return render_template('hide/hide.html')


@hide_bp.route('/process', methods=['POST'])
@login_required
def process():
    try:
        cover_file  = request.files.get('cover_image')
        password    = request.form.get('password', '').strip()
        secret_type = request.form.get('secret_type', 'text')
        cover_mode  = request.form.get('cover_mode', 'Universal')

        if not cover_file or not cover_file.filename:
            return jsonify({'error': 'No cover file provided.'}), 400
        if len(password) < 4:
            return jsonify({'error': 'Password must be at least 4 characters.'}), 400

        cover_ext = _ext(cover_file.filename)
        cover_is_3d = cover_ext in ALLOWED_COVER_3D
        cover_is_img = cover_ext in ALLOWED_COVER_IMG

        if not cover_is_3d and not cover_is_img:
            return jsonify({'error': 'Unsupported cover format. Use PNG/JPG or .obj/.npy/.bin.'}), 400

        # ── Read cover bytes ──────────────────────────────────────────────────
        cover_bytes = cover_file.read()

        # ── Prepare secret payload ────────────────────────────────────────────
        if secret_type == 'text':
            text = request.form.get('secret_text', '').strip()
            if not text:
                return jsonify({'error': 'Secret text is empty.'}), 400
            secret_bytes = secret_text_to_bytes(text)

        elif secret_type == 'image':
            sf = request.files.get('secret_image')
            if not sf or not _allowed(sf.filename, ALLOWED_SECRET_IMG):
                return jsonify({'error': 'Please upload a valid secret image.'}), 400
            secret_bytes = secret_image_to_bytes(sf.stream)

        elif secret_type == '3d':
            sf = request.files.get('secret_3d')
            if not sf:
                return jsonify({'error': 'Please upload a 3D data file.'}), 400
            secret_bytes = secret_3d_to_bytes(sf.stream)

        else:
            return jsonify({'error': 'Unknown secret type.'}), 400

        # ── Embed ─────────────────────────────────────────────────────────────
        if cover_is_img:
            cover_img = Image.open(io.BytesIO(cover_bytes))
            stego_img, psnr, ssim = neural_embed(cover_img, secret_bytes, password)
            robustness = compute_robustness_score(psnr, ssim)

            buf = io.BytesIO()
            stego_img.save(buf, format='PNG', optimize=False)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()

            result = {
                'success': True,
                'cover_type': 'image',
                'stego_image': f'data:image/png;base64,{b64}',
                'stego_filename': 'stego_' + os.path.splitext(cover_file.filename)[0] + '.png',
                'psnr': psnr,
                'ssim': ssim,
                'robustness': robustness,
                'model_used': f'{cover_mode} (Neural AI U-Net)',
                'secret_type': secret_type,
            }

        else:  # 3D cover
            stego_bytes = embed_in_3d(cover_bytes, secret_bytes, password, cover_file.filename)
            b64 = base64.b64encode(stego_bytes).decode()
            stego_fname = 'stego_' + cover_file.filename

            result = {
                'success': True,
                'cover_type': '3d',
                'stego_3d_data': b64,
                'stego_filename': stego_fname,
                'stego_ext': cover_ext,
                'psnr': '—',
                'ssim': '—',
                'robustness': '—',
                'model_used': f'3D Neural Cover AI ({cover_ext.upper()})',
                'secret_type': secret_type,
            }

        # ── Save History ──────────────────────────────────────────────────────
        entry = History(
            user_id=current_user.id,
            operation='hide',
            file_name=cover_file.filename,
            model_used=result['model_used'],
            psnr_value=psnr if cover_is_img else None,
            robustness_score=robustness if cover_is_img else None,
            result_type=secret_type,
        )
        db.session.add(entry)
        db.session.commit()

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
