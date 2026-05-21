from settings import FILE_BEST_MODEL, MODELS_DIR


def model_ready():
    return (MODELS_DIR / FILE_BEST_MODEL).is_file()
