from flask import Blueprint, render_template, current_app

bp = Blueprint('dashboard', __name__)


@bp.route('/')
def index():
    return render_template('dashboard.html', title='Dashboard')