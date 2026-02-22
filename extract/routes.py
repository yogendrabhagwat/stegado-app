import io
from flask import render_template, request, jsonify
from flask_login import login_required, current_user
from extract import extract_bp
from extensions import db
from models import History
from utils.neural_stego import neural_extract, bytes_to_secret
from utils.cover_3d import extract_from_3d, is_3d_file
from PIL import Image

ALLOWED_STEGO_IMG = {'png'}
ALLOWED_STEGO_3D  = {'obj', 'npy', 'npz', 'bin', 'ply', 'stl', 'glb', 'fbx'}


def _ext(filename: str) -> str:
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''


@extract_bp.route('/')
@login_required
def extract_page():
    return render_template('extract/extract.html')


@extract_bp.route('/process', methods=['POST'])
@login_required
def process():
    try:
        stego_file = request.files.get('stego_image')
        password   = request.form.get('password', '').strip()

        if not stego_file or not stego_file.filename:
            return jsonify({'error': 'No stego file uploaded.'}), 400
        if len(password) < 4:
            return jsonify({'error': 'Password must be at least 4 characters.'}), 400

        ext = _ext(stego_file.filename)
        stego_bytes = stego_file.read()

        if ext in ALLOWED_STEGO_3D:
            raw_bytes = extract_from_3d(stego_bytes, password, stego_file.filename)
        elif ext in ALLOWED_STEGO_IMG:
            stego_img = Image.open(io.BytesIO(stego_bytes))
            raw_bytes = neural_extract(stego_img, password)
        else:
            return jsonify({'error': f'Unsupported file type ".{ext}". Upload a PNG stego image or stego 3D file.'}), 400

        result = bytes_to_secret(raw_bytes)

        # Save history
        try:
            model_label = f'Neural AI ({"3D Cover" if ext in ALLOWED_STEGO_3D else "Image Cover"})'
        except Exception:
            model_label = 'Neural AI'

        entry = History(
            user_id=current_user.id,
            operation='extract',
            file_name=stego_file.filename,
            model_used=model_label,
            result_type=result.get('type', 'text'),
        )
        db.session.add(entry)
        db.session.commit()

        return jsonify({'success': True, 'result': result})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Extraction failed: {str(e)}'}), 400
