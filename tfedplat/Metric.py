


class Metric:
    def __init__(self, name='Metric', *args, **kwargs):
        self.name = name

    @staticmethod
    def calc(network_output, target):
        raise NotImplementedError
