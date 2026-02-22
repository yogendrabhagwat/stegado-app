from flask import Blueprint
extract_bp = Blueprint('extract', __name__, url_prefix='/extract', template_folder='../templates/extract')
from extract import routes  # noqa: F401, E402
