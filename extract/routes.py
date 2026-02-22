from flask import render_template, request, jsonify
from flask_login import login_required, current_user
from extract import extract_bp
from extensions import db
from models import History
from utils.neural_stego import neural_extract, bytes_to_secret
from PIL import Image


@extract_bp.route('/')
@login_required
def extract_page():
    return render_template('extract/extract.html')


@extract_bp.route('/process', methods=['POST'])
@login_required
def process():
    try:
        stego_file = request.files.get('stego_image')
        password = request.form.get('password', '').strip()

        if not stego_file or not stego_file.filename:
            return jsonify({'error': 'No stego image uploaded.'}), 400
        if not stego_file.filename.lower().endswith('.png'):
            return jsonify({'error': 'Please upload a PNG stego image.'}), 400
        if len(password) < 4:
            return jsonify({'error': 'Password must be at least 4 characters.'}), 400

        stego_img = Image.open(stego_file.stream)
        raw_bytes = neural_extract(stego_img, password)
        result = bytes_to_secret(raw_bytes)

        # Save history
        try:
            from core.model_loader import get_active_model
            model_label = get_active_model('Universal')
        except Exception:
            model_label = 'Neural Fourier AI'

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
        return jsonify({'error': f'Extraction failed: {str(e)}'}), 400
