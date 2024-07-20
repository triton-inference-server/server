from dataclasses import dataclass

@dataclass
class KServeHttpOptions():
    address: str
    port: int
    thread_cnt: int
    restricted_protocols: list