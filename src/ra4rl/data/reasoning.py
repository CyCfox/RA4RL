"""Explicit provider thinking parameters; legacy configurations stay disabled."""


def reasoning_options(cfg):
    thinking = cfg.get('thinking', {'type': 'disabled'})
    if (not isinstance(thinking, dict) or set(thinking) != {'type'}
            or thinking['type'] not in ('enabled', 'disabled')):
        raise ValueError('thinking must contain only type=enabled or type=disabled')
    options = {'thinking': dict(thinking)}
    effort = cfg.get('reasoning_effort')
    if effort is not None:
        if thinking['type'] != 'enabled' or effort not in ('low', 'high', 'max'):
            raise ValueError('reasoning_effort requires enabled thinking and low/high/max')
        options['reasoning_effort'] = effort
    return options
