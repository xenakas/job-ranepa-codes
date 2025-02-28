from utils_d.flask_wrapper import flask_wrapper
# from utils_d.flask_wrapper import  create_supervisord_conf
from characteristics_extractor import CharactericsExtractor

extractor = CharactericsExtractor()


def parse_text(text):
    result = extractor.parse_text([text], piped=False)
    return result


# create_supervisord_conf(env="products", port=5011)
flask_wrapper(parse_text)
