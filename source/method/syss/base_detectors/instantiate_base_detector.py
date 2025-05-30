from source.method.syss.base_detectors.shi_tomasi import ShiTomasi


def instantiate_base_detector(cfg):
    if cfg.name == 'shi_tomasi':
        return ShiTomasi.from_config(cfg)
    
    else:
        raise ValueError(f"Unknown base detector {cfg.name}")
