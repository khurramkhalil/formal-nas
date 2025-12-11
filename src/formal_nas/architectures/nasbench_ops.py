
import torch
import torch.nn as nn

# Operations Map
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'nor_conv_3x3': lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 1, affine=affine),
    'nor_conv_1x1': lambda C, stride, affine: ReLUConvBN(C, C, 1, stride, 0, affine=affine),
    'skip_connect': lambda C, stride, affine: nn.Dropout(0.0) if stride == 1 else FactorizedReduce(C, C, affine=affine),
}

# TransNAS Specific (Mapping to above or custom)
# TransNAS 'rcb_3x3' is roughly 'nor_conv_3x3'
# TransNAS 'zero' is 'none'
# TransNAS 'identity' is 'skip_connect'

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    def forward(self, x):
        return self.op(x)

class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
    def forward(self, x):
        # HLS4ML-Friendly Zero: Avoid slicing if possible, or just multiply.
        # Ideally, hls4ml optimizes x*0 to constant 0.
        if self.stride == 1:
            return x * 0.0
        # For stride > 1, we must subsample. 
        # But in this NAS-Bench-201 Cell, stride is ALWAYS 1.
        return x * 0.0

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        # Simplified for HLS: FactorizedReduce is complex. 
        # Since stride=1 in our use case, this class is UNUSED in the forward pass.
        # But we define it safely just in case.
        self.conv = nn.Conv2d(C_in, C_out, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        # We replace the complex dual-path conv with a simple strided conv for safety
        # because this code should NOT be reached in stride=1 cells.
        return self.bn(self.conv(x))

class Cell(nn.Module):
    def __init__(self, op_indices, C_in, C_out, stride, affine=True):
        super(Cell, self).__init__()
        self.op_names = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
        self.edges = nn.ModuleList()
        for op_idx in op_indices:
            op_name = self.op_names[op_idx]
            self.edges.append(OPS[op_name](C_in, stride, affine))
            
    def forward(self, x):
        node0 = x
        node1 = self.edges[0](node0)
        node2 = self.edges[1](node0) + self.edges[3](node1)
        node3 = self.edges[2](node0) + self.edges[4](node1) + self.edges[5](node2)
        return node3

class TinyNetwork(nn.Module):
    def __init__(self, op_indices, C=16, num_classes=10):
        super(TinyNetwork, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        self.cell = Cell(op_indices, C, C, 1, affine=True)
        
        # HLS-Friendly Global Pooling
        # Explicit kernel/stride to ensure fixed size output 1x1
        self.global_pool = nn.AvgPool2d(kernel_size=32, stride=32)
        self.classifier = nn.Linear(C, num_classes)
        
    def forward(self, x):
        out = self.stem(x)
        out = self.cell(out)
        out = self.global_pool(out)
        out = out.flatten(1)
        out = self.classifier(out)
        return out
