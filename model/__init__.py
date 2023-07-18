def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    return m
