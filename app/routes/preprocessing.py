from flask import Blueprint, render_template

bp = Blueprint('preprocessing', __name__)


@bp.route('/preprocessing')
def index():
    return render_template('preprocessing.html', title='Preprocessing Data')