from flask import Blueprint, render_template

bp = Blueprint('eda', __name__)


@bp.route('/eda')
def index():
    return render_template('eda.html', title='Exploratory Data Analysis')