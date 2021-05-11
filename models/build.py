from .vit import ViT

def build_model(config):
    model_type = config.MODEL['TYPE']
    if model_type == 'vit':
        model = ViT(image_size=config.IMG_SIZE,
                    patch_size=config.MODEL['VIT']['PATCH_SIZE'],
                    num_classes=config.MODEL['VIT']['NUM_CLASSES'],
                    dim=config.MODEL['VIT']['DIM'],
                    depth=config.MODEL['VIT']['DEPTH'],
                    heads=config.MODEL['VIT']['HEADS'],
                    mlp_dim=config.MODEL['VIT']['MLP_DIM'],
                    dropout=config.MODEL['VIT']['DROPOUT'],
                    emb_dropout=config.MODEL['VIT']['EMB_DROPOUT'],
                    pool=config.MODEL['VIT']['POOL'])
    else:
        raise NotImplementedError(f"Unknow model: {model_type}")

    return model