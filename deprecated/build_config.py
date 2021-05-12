from types import MethodType
import os
from datetime import datetime

class Configs:
    def __init__(self):
        self.TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)
        return args_dict

    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def __str__(self):
        # print Hyper Parameters
        settings_str = ''
        for attr in dir(self):
            # 如果不加 attr.startwith('__')会打印出很多额外的参数，是自身自带的一些默认方法和属性
            if not 'np' in attr and not 'random' in attr and not attr.startswith('__') and not isinstance(
                    getattr(self, attr), MethodType):
                settings_str += '{ %-17s }->' % attr + str(getattr(self, attr)) + '\n'
        return settings_str

configs = Configs()
