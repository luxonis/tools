from torch import nn


class EfficientRepV2(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(
        self,
        old_efficientrep
    ):
        super().__init__()

        self.fuse_P2 = old_efficientrep.fuse_P2 if hasattr(old_efficientrep, 'fuse_P2') else False

        self.stem = old_efficientrep.stem
        self.ERBlock_2 = old_efficientrep.ERBlock_2
        self.ERBlock_3 = old_efficientrep.ERBlock_3
        self.ERBlock_4 = old_efficientrep.ERBlock_4
        self.ERBlock_5 = old_efficientrep.ERBlock_5

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)


class EfficientRep6V2(nn.Module):
    '''EfficientRep+P6 Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(
        self,
        old_layer
    ):
        super().__init__()

        self.fuse_P2 = old_layer.fuse_P2 if hasattr(old_layer, 'fuse_P2') else False

        self.stem = old_layer.stem
        self.ERBlock_2 = old_layer.ERBlock_2
        self.ERBlock_3 = old_layer.ERBlock_3
        self.ERBlock_4 = old_layer.ERBlock_4
        self.ERBlock_5 = old_layer.ERBlock_5
        self.ERBlock_6 = old_layer.ERBlock_6

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)

        return tuple(outputs)


class CSPBepBackboneV2(nn.Module):
    """
    CSPBepBackbone module.
    """

    def __init__(
        self,
        old_layer
    ):
        super().__init__()

        self.fuse_P2 = old_layer.fuse_P2 if hasattr(old_layer, 'fuse_P2') else False

        self.stem = old_layer.stem
        self.ERBlock_2 = old_layer.ERBlock_2
        self.ERBlock_3 = old_layer.ERBlock_3
        self.ERBlock_4 = old_layer.ERBlock_4
        self.ERBlock_5 = old_layer.ERBlock_5

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)


class CSPBepBackbone_P6V2(nn.Module):
    """
    CSPBepBackbone+P6 module. 
    """

    def __init__(
        self,
        old_layer
    ):
        super().__init__()

        self.fuse_P2 = old_layer.fuse_P2 if hasattr(old_layer, 'fuse_P2') else False

        self.stem = old_layer.stem
        self.ERBlock_2 = old_layer.ERBlock_2
        self.ERBlock_3 = old_layer.ERBlock_3
        self.ERBlock_4 = old_layer.ERBlock_4
        self.ERBlock_5 = old_layer.ERBlock_5
        self.ERBlock_6 = old_layer.ERBlock_6

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)

        return tuple(outputs)
    