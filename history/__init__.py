from flask import Blueprint
history_bp = Blueprint('history', __name__, url_prefix='/history', template_folder='../templates/history')
from history import routes  # noqa: F401, E402
