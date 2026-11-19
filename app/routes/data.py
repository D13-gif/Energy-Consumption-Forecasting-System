from flask import Blueprint, render_template

bp = Blueprint('data', __name__)


@bp.route('/data')
def index():
    return render_template('data.html', title='Data Explorer')