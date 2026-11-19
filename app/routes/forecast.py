from flask import Blueprint, render_template

bp = Blueprint('forecast', __name__)


@bp.route('/forecast')
def index():
    return render_template('forecast.html', title='Forecast')