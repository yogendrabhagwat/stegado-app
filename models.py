from extensions import db
from flask_login import UserMixin
from datetime import datetime


class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email_or_mobile = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    history = db.relationship('History', backref='user', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User {self.username}>'


class History(db.Model):
    __tablename__ = 'history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    operation = db.Column(db.String(20), nullable=False)   # hide / extract
    file_name = db.Column(db.String(256))
    model_used = db.Column(db.String(50))                  # Universal / 2D / 3D
    psnr_value = db.Column(db.Float)
    robustness_score = db.Column(db.Float)
    result_type = db.Column(db.String(20))                 # text / image / 3d
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<History {self.operation} {self.timestamp}>'
