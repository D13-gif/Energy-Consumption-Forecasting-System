import os
from dotenv import load_dotenv

load_dotenv()


def create_application():
    from app import create_app

    return create_app(os.environ.get('FLASK_CONFIG', 'development'))


app = create_application()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.debug)