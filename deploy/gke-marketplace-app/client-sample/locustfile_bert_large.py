# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from locust import HttpUser, task, between
from locust import LoadTestShape
import json


class ProfileLoad(LoadTestShape):
    '''
    This load profile starts at 0 and steps up by step_users
    increments every tick, up to target_users.  After reaching
    target_user level, load will stay at target_user level
    until time_limit is reached.
    '''

    target_users = 1000
    step_users = 50  # ramp users each step
    time_limit = 3600  # seconds

    def tick(self):
        num_steps = self.target_users / self.step_users
        run_time = round(self.get_run_time())

        if run_time < self.time_limit:
            if num_steps < run_time:
                user_count = num_steps * self.step_users
            else:
                user_count = self.target_users
            return (user_count, self.step_users)
        else:
            return None


class TritonUser(HttpUser):
    wait_time = between(0.2, 0.2)

    @task()
    def bert(self):
        response = self.client.post(self.url1, data=json.dumps(self.data))

    def on_start(self):
        with open('bert_request.json') as f:
            self.data = json.load(f)

        self.url1 = '{}/v2/models/{}/infer'.format(self.environment.host,
                                                   'bert_large')
