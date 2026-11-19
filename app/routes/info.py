from flask import Blueprint, render_template

bp = Blueprint('info', __name__)


@bp.route('/model-info')
def model_info():
    return render_template('model_info.html', title='Model Information')


@bp.route('/about')
def about():
    return render_template('about.html', title='Tentang EnergiCast')