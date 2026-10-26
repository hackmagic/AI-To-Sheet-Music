"""render 层（v2 架构）：绝对时间轴、ABC 兜底谱与网页渲染载荷。

实现见同目录各模块；对外契约见 :mod:`aitsm.contracts`。

* :func:`aitsm.render.timing.build_timing` —— beat -> 绝对秒；
* :func:`aitsm.render.abc.write_abc` —— 可读 ABC（VexFlow 失败时给 abcjs）；
* :func:`aitsm.render.payload.build_payload` —— 网页渲染用的唯一 JSON。

三个入口都是纯函数，缺失字段留空，不猜数据。
"""

from .abc import write_abc
from .payload import PAYLOAD_SCHEMA, build_payload
from .timing import build_timing, parse_duration

__all__ = [
    "PAYLOAD_SCHEMA",
    "build_payload",
    "build_timing",
    "parse_duration",
    "write_abc",
]
