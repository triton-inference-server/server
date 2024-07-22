from dataclasses import dataclass

@dataclass
class KServeHttpOptions():
    address: str
    port: int
    thread_count: int
    restricted_protocols: list