import asyncio

import cupy
import numpy
import pytest
import tritonserver


@pytest.fixture(autouse=True)
def add_tritonserver(doctest_namespace):
    doctest_namespace["tritonserver"] = tritonserver
    doctest_namespace["numpy"] = numpy
    doctest_namespace["cupy"] = cupy
    doctest_namespace["asyncio"] = asyncio
