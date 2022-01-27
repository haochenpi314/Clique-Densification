import click


def add_to_click_cli(command_group, func):
    import inspect
    print(inspect.getfullargspec(func))
    # print(func.__annotations__)
    annotations = inspect.getfullargspec(func).annotations
    defaults = inspect.getfullargspec(func).defaults
    defaults = list(reversed(defaults)) if defaults is not None else []

    def decorator(func):
        def wrapper(f):
            for i, para_name in enumerate(reversed(inspect.getfullargspec(func).args)):
                attrs = {}
                if para_name in annotations:
                    attrs = annotations[para_name]
                else:
                    if i < len(defaults):
                        attrs = {'default': defaults[i]}

                param_decls = ('--%s' % para_name,)
                click.option(*param_decls, **attrs)(f)

            return f
        return wrapper

    @command_group.command(name='%s' % func.__name__)
    @decorator(func)
    def temp(*args, **kwargs):
        func(*args, **kwargs)
