from dataclasses import dataclass
from typing import Any, Union
from enum import Enum


class Validation:
    def validate_type(self, value: Any, expected_type: Any, param_name: str):
        if not isinstance(value, expected_type):
            raise TypeError(f"Incorrect Type for {param_name}. Expected {expected_type}, got {type(value)}")
    
    #TODO: implement to catch ints in python that are too big for int32_t in C++
    #TODO: Make sure port >= 0. Cannot have negative port.
    def validate_range(self, value, lb, ub, param_name):
        pass
    
    def validate(self):
        for param_name, param_type in self.__annotations__.items():
            value = getattr(self, param_name)
            self.validate_type(value, param_type, param_name)










    