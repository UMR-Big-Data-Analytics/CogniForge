from configparser import ConfigParser


def _parse_range(input: str) -> range:
    min, max = input.split('-')
    return range(
        int(min.strip()),
        int(max.strip())
    )


_config = ConfigParser(converters={
    'range': _parse_range
})

_config.read('config.ini')
_config.read('secrets.ini')

FURTHR_MIND = _config['FurthrMind']
MACHINE_LEARNING = _config['MachineLearning']