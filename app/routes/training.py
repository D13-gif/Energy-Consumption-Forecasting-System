from flask import Blueprint, render_template

bp = Blueprint('training', __name__)


@bp.route('/model-training')
def index():
    return render_template('training.html', title='Model Training')