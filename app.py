import os
import logging
from flask import Flask, render_template
from extensions import db, bcrypt, login_manager, csrf


def create_app():
    app = Flask(__name__)

    # ── Config ────────────────────────────────────────────────────────────────
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'stegano-secret-change-me-in-prod-2024'),
        SQLALCHEMY_DATABASE_URI='sqlite:///database.db',
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MAX_CONTENT_LENGTH=32 * 1024 * 1024,   # 32 MB upload limit
        WTF_CSRF_ENABLED=True,
        WTF_CSRF_TIME_LIMIT=None,
    )

    # ── Extensions ────────────────────────────────────────────────────────────
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        from models import User
        return User.query.get(int(user_id))

    # ── Blueprints ────────────────────────────────────────────────────────────
    from auth import auth_bp
    from hide import hide_bp
    from extract import extract_bp
    from history import history_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(hide_bp)
    app.register_blueprint(extract_bp)
    app.register_blueprint(history_bp)

    # ── CSRF exempt API routes ────────────────────────────────────────────────
    csrf.exempt(hide_bp)
    csrf.exempt(extract_bp)

    # ── Index route ───────────────────────────────────────────────────────────
    @app.route('/')
    def index():
        return render_template('index.html')

    # ── Error handlers ────────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return render_template('errors/404.html'), 404

    @app.errorhandler(413)
    def too_large(e):
        return render_template('errors/413.html'), 413

    @app.errorhandler(500)
    def server_error(e):
        return render_template('errors/500.html'), 500

    # ── Init DB ───────────────────────────────────────────────────────────────
    with app.app_context():
        db.create_all()

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    return app


if __name__ == '__main__':
    application = create_app()
    application.run(debug=True, host='0.0.0.0', port=5000)
