# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def raise_error(msg):
    """
    Raise error with the provided message
    """
    raise InferenceServerException(msg=msg) from None


class InferenceServerException(Exception):
    """Exception indicating non-Success status.

    Parameters
    ----------
    err : RequestStatus Protobuf
        The protobuf message describing the error

    """

    def __init__(self, msg, status=None, debug_details=None):
        self._msg = msg
        self._status = status
        self._debug_details = debug_details

    def __str__(self):
        msg = super().__str__() if self._msg is None else self._msg
        if self._status is not None:
            msg = '[' + self._status + '] ' + msg
        return msg

    def message(self):
        """Get the exception message.

        Returns
        -------
        str
            The message associated with this exception, or None if no message.

        """
        return self._msg

    def status(self):
        """Get the status of the exception.

        Returns
        -------
        str
            Returns the status of the exception

        """
        return self._status

    def debug_details(self):
        """Get the detailed information about the exception for debugging purposes

        Returns
        -------
        str
            Returns string in JSON format containing the exception details            

        """
        return self._debug_details
