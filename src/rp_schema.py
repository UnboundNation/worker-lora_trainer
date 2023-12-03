INPUT_SCHEMA = {
    'zip_url': {
        'type': str,
        'required': True
    },
    'instance_name': {
        'type': str,
        'required': True
    },
    'class_name': {
        'type': str,
        'required': True
    },
    'steps': {
        'type': int,
        'required': False,
        'default': 30
    }
}
