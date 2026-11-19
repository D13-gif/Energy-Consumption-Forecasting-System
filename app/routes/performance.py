from flask import Blueprint, render_template

bp = Blueprint('performance', __name__)


@bp.route('/model-performance')
def index():
    return render_template('performance.html', title='Model Performance')