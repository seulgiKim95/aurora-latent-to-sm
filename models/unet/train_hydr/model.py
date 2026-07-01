from train_hydr.aurora_lite import AuroraPretrained
from train_hydr.aurora_lite import AuroraSmallPretrained

from train_hydr import AuroraDataset
from train_hydr import locations
from train_hydr import scales
from train_hydr.data import AuroraDatasetLiteDecoder
from train_hydr.decoder import Decoder


def load_aurora_lite_model():
    """load Pretrained model"""
    model = AuroraPretrained()
    model.load_checkpoint()
    for name, param in model.named_parameters(): # freeze
        param.requires_grad = False
    return model


def load_aurora_lite_small_model():
    """load small pretrained model"""
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    for name, param in model.named_parameters(): # freeze
        param.requires_grad = False
    return model


def get_new_variable_parameters(model, new_vars):
    if isinstance(new_vars, str):
        new_vars = [new_vars]
    new_params = []
    exist_params = []
    for name, param in model.named_parameters():
        if any([var in name for var in new_vars]) and param.requires_grad:
            new_params.append(param)
        elif param.requires_grad:
            exist_params.append(param)
        else:
            pass
    return new_params, exist_params


def load_decoder(model_aurora):
    new_vars = AuroraDatasetLiteDecoder.new_vars
    P = model_aurora.decoder.patch_size
    E = model_aurora.encoder.embed_dim
    model = Decoder(
        surf_vars_new = new_vars,
        embed_dim = E, # input layer
    )
    return model
