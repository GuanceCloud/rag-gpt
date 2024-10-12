from functools import wraps
from flask import Blueprint, request
from server.app.utils.token_helper import TokenHelper
from server.logger.logger_config import my_logger as logger
import os 

auth_bp = Blueprint('auth', __name__, url_prefix='/open_kf_api/auth')

guance_api_secret_key = os.environ.get('GUANCE_SECRET', "") or ""
if guance_api_secret_key == "":
    logger.warning('GUANCE_SECRET is ""')


def check_guance_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        cur_api_key = request.headers.get("GUANCE-API-KEY", "")
        if cur_api_key != guance_api_secret_key:
            logger.error(f"guance api key is invalid")
            return {
                'retcode': -40001,
                'message': 'guance api key is invalid',
                'data': {}
            }, 401

        return f(*args, **kwargs)

    return decorated_function

@auth_bp.route('/get_token', methods=['POST'])
@check_guance_api_key
def get_token():
    data = request.json
    user_id = data.get('user_id')
    if not user_id:
        return {
            'retcode': -20000,
            'message': 'user_id is required',
            'data': {}
        }

    try:
        # generate token
        token = TokenHelper.generate_token(user_id)
        logger.success(f"Generate token: '{token}' with user_id: '{user_id}'")
        return {"retcode": 0, "message": "success", "data": {"token": token}}
    except Exception as e:
        logger.error(
            f"Generate token with user_id: '{user_id}' is failed, the exception is {e}"
        )
        return {'retcode': -20001, 'message': str(e), 'data': {}}
