from flask import render_template, jsonify
from flask_login import login_required, current_user
from history import history_bp
from models import History


@history_bp.route('/')
@login_required
def history_page():
    records = (History.query
               .filter_by(user_id=current_user.id)
               .order_by(History.timestamp.desc())
               .limit(100)
               .all())
    return render_template('history/history.html', records=records)


@history_bp.route('/api')
@login_required
def history_api():
    records = (History.query
               .filter_by(user_id=current_user.id)
               .order_by(History.timestamp.desc())
               .limit(100)
               .all())
    data = [{
        'id': r.id,
        'operation': r.operation,
        'file_name': r.file_name,
        'model_used': r.model_used,
        'psnr_value': r.psnr_value,
        'robustness_score': r.robustness_score,
        'result_type': r.result_type,
        'timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M'),
    } for r in records]
    return jsonify(data)
