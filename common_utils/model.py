from aurora import AuroraPretrained
from aurora import AuroraSmallPretrained


def load_aurora_pretrained_model():
    """load Pretrained model"""
    model = AuroraPretrained()
    model.load_checkpoint()
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model


def load_aurora_pretrained_small_model():
    """load small pretrained model"""
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model


def print_trainable_params(model):
    """print the number of trainable parameters"""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return trainable_params
