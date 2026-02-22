from flask import Blueprint
hide_bp = Blueprint('hide', __name__, url_prefix='/hide', template_folder='../templates/hide')
from hide import routes  # noqa: F401, E402
