"""时序分割模型注册表。"""

from .asformer import ASFormerLite
from .bigru import BiGRU
from .ms_tcn import MSTCN

MODEL_REGISTRY = {
    "ms_tcn": MSTCN,
    "asformer": ASFormerLite,
    "bigru": BiGRU,
}

MODEL_PRIORITY = {
    "ms_tcn": 2,
    "asformer": 1,
    "bigru": 0,
}

