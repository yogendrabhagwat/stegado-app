from flask import Blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth', template_folder='../templates/auth')
from auth import routes  # noqa: F401, E402
