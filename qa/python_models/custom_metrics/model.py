# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import numpy as np
import requests
import triton_python_backend_utils as pb_utils


class PBCustomMetricsTest(unittest.TestCase):
    def _get_metrics(self):
        metrics_url = "http://localhost:8002/metrics"
        r = requests.get(metrics_url)
        r.raise_for_status()
        return r.text

    def _metric_api_helper(self, metric, kind):
        # Adding logger to test if custom metrics and logging work together
        # as they use the same message queue.
        logger = pb_utils.Logger

        # The value should be 0.0 before the test
        self.assertEqual(metric.value(), 0.0)

        # Test increment positive value
        increment = 2023.0
        metric.increment(increment)
        self.assertEqual(metric.value(), increment)
        logger.log_info("Incremented metric to : {}".format(metric.value()))

        # Test increment negative value
        decrement = -23.5
        if kind == "counter":
            # Counter should not accept negative values
            with self.assertRaises(pb_utils.TritonModelException):
                metric.increment(decrement)
        else:
            metric.increment(decrement)
            self.assertEqual(metric.value(), increment + decrement)
            logger.log_info("Decremented metric to : {}".format(metric.value()))

        # Test set value
        value = 999.9
        if kind == "counter":
            # Counter does not support set
            with self.assertRaises(pb_utils.TritonModelException):
                metric.set(value)
        else:
            metric.set(value)
            self.assertEqual(metric.value(), value)
            logger.log_info("Set metric to : {}".format(metric.value()))

        # Test observe value
        observe = 0.05
        # Counter and gauge do not support observe
        with self.assertRaises(pb_utils.TritonModelException):
            metric.observe(observe)

    def _histogram_api_helper(self, metric, name, labels):
        def histogram_str_builder(name, type, labels, value, le=None):
            if type == "count" or type == "sum":
                return f"{name}_{type}{{{labels}}} {value}"
            elif type == "bucket":
                return f'{name}_bucket{{{labels},le="{le}"}} {value}'
            else:
                raise

        # Adding logger to test if custom metrics and logging work together
        # as they use the same message queue.
        logger = pb_utils.Logger

        # All values should be 0.0 before the test
        metrics = self._get_metrics()
        self.assertIn(histogram_str_builder(name, "count", labels, "0"), metrics)
        self.assertIn(histogram_str_builder(name, "sum", labels, "0"), metrics)
        self.assertIn(
            histogram_str_builder(name, "bucket", labels, "0", le="0.1"), metrics
        )
        self.assertIn(
            histogram_str_builder(name, "bucket", labels, "0", le="1"), metrics
        )
        self.assertIn(
            histogram_str_builder(name, "bucket", labels, "0", le="2.5"), metrics
        )
        self.assertIn(
            histogram_str_builder(name, "bucket", labels, "0", le="5"), metrics
        )
        self.assertIn(
            histogram_str_builder(name, "bucket", labels, "0", le="10"), metrics
        )
        self.assertIn(
            histogram_str_builder(name, "bucket", labels, "0", le="+Inf"), metrics
        )

        # Histogram does not support value
        with self.assertRaises(pb_utils.TritonModelException):
            metric.value()

        # Test increment value
        increment = 2023.0
        # Histogram does not support increment
        with self.assertRaises(pb_utils.TritonModelException):
            metric.increment(increment)

        # Test set value
        value = 999.9
        # Histogram does not support set
        with self.assertRaises(pb_utils.TritonModelException):
            metric.set(value)

        # Test observe value
        data = [0.05, 1.5, 6.0]
        for datum in data:
            metric.observe(datum)
            logger.log_info("Observe histogram metric with value : {}".format(datum))

        metrics = self._get_metrics()
        self.assertIn(
            histogram_str_builder(name, "count", labels, str(len(data))), metrics
        )
        self.assertIn(
            histogram_str_builder(name, "sum", labels, str(sum(data))), metrics
        )
        self.assertIn(
            histogram_str_builder(name, "bucket", labels, "1", le="0.1"), metrics
        )
        self.assertIn(
            histogram_str_builder(name, "bucket", labels, "1", le="1"), metrics
        )
        self.assertIn(
            histogram_str_builder(name, "bucket", labels, "2", le="2.5"), metrics
        )
        self.assertIn(
            histogram_str_builder(name, "bucket", labels, "2", le="5"), metrics
        )
        self.assertIn(
            histogram_str_builder(name, "bucket", labels, "3", le="10"), metrics
        )
        self.assertIn(
            histogram_str_builder(name, "bucket", labels, "3", le="+Inf"), metrics
        )

    def _dup_metric_helper(self, labels={}):
        # Adding logger to test if custom metrics and logging work together
        # as they use the same message queue.
        logger = pb_utils.Logger

        description = "dup metric"
        metric_family = pb_utils.MetricFamily(
            name="test_dup_metric",
            description=description,
            kind=pb_utils.MetricFamily.COUNTER,
        )

        # Verify dupe metrics reference same underlying metric
        metric1 = metric_family.Metric(labels=labels)
        metric2 = metric_family.Metric(labels=labels)

        # The value should be 0 before the test
        self.assertEqual(metric1.value(), 0.0)
        self.assertEqual(metric2.value(), 0.0)

        # Increment metric 1, check metric 2 == metric 1
        increment = 7.5
        metric1.increment(increment)
        self.assertEqual(metric1.value(), metric2.value())
        logger.log_info("Incremented metric1 to : {}".format(metric1.value()))
        logger.log_info("Incremented metric2 to : {}".format(metric2.value()))

        # Assert custom metric/family remains when there's still a reference to it
        del metric1
        metrics = self._get_metrics()
        self.assertIn(description, metrics)

    def test_counter_e2e(self):
        metric_family = pb_utils.MetricFamily(
            name="test_counter_e2e",
            description="test metric counter kind end to end",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        labels = {"example1": "counter_label1", "example2": "counter_label2"}
        metric = metric_family.Metric(labels=labels)
        self._metric_api_helper(metric, "counter")

        pattern = (
            'test_counter_e2e{example1="counter_label1",example2="counter_label2"}'
        )
        metrics = self._get_metrics()
        self.assertIn(pattern, metrics)

    def test_gauge_e2e(self):
        metric_family = pb_utils.MetricFamily(
            name="test_gauge_e2e",
            description="test metric gauge kind end to end",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        labels = {"example1": "gauge_label1", "example2": "gauge_label2"}
        metric = metric_family.Metric(labels=labels)
        self._metric_api_helper(metric, "gauge")

        pattern = 'test_gauge_e2e{example1="gauge_label1",example2="gauge_label2"}'
        metrics = self._get_metrics()
        self.assertIn(pattern, metrics)

    def test_histogram_e2e(self):
        name = "test_histogram_e2e"
        metric_family = pb_utils.MetricFamily(
            name=name,
            description="test metric histogram kind end to end",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )

        labels = {"example1": "histogram_label1", "example2": "histogram_label2"}
        buckets = [0.1, 1.0, 2.5, 5.0, 10.0]
        metric = metric_family.Metric(labels=labels, buckets=buckets)

        labels_str = 'example1="histogram_label1",example2="histogram_label2"'
        self._histogram_api_helper(metric, name, labels_str)

        metrics = self._get_metrics()
        count_pattern = f"{name}_count{{{labels_str}}}"
        sum_pattern = f"{name}_sum{{{labels_str}}}"
        bucket_pattern = f"{name}_bucket{{{labels_str}"
        self.assertEqual(metrics.count(count_pattern), 1)
        self.assertEqual(metrics.count(sum_pattern), 1)
        self.assertEqual(metrics.count(bucket_pattern), len(buckets) + 1)

    def test_histogram_args(self):
        name = "test_histogram_args"
        metric_family = pb_utils.MetricFamily(
            name=name,
            description="test metric histogram args",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )

        # Test "None" value buckets
        with self.assertRaises(pb_utils.TritonModelException):
            metric_family.Metric(labels={})
        with self.assertRaises(pb_utils.TritonModelException):
            metric_family.Metric(labels={}, buckets=None)

        # Test non-ascending order buckets
        with self.assertRaises(pb_utils.TritonModelException):
            metric_family.Metric(labels={}, buckets=[2.5, 0.1, 1.0, 10.0, 5.0])

        # Test duplicate value buckets
        with self.assertRaises(pb_utils.TritonModelException):
            metric_family.Metric(labels={}, buckets=[1, 1, 2, 5, 5])

        # Test empty list bucket
        metric_family.Metric(labels={}, buckets=[])

    def test_dup_metric_family_diff_kind(self):
        # Test that a duplicate metric family can't be added with a conflicting type/kind
        metric_family1 = pb_utils.MetricFamily(
            name="test_dup_metric_family_diff_kind",
            description="test metric family with same name but different kind",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        with self.assertRaises(pb_utils.TritonModelException):
            metric_family2 = pb_utils.MetricFamily(
                name="test_dup_metric_family_diff_kind",
                description="test metric family with same name but different kind",
                kind=pb_utils.MetricFamily.GAUGE,
            )
            self.assertIsNone(metric_family2)

        self.assertIsNotNone(metric_family1)

    def test_dup_metric_family_diff_description(self):
        # Test that a duplicate metric family name will still return the
        # original metric family even if the description is changed
        metric_family1 = pb_utils.MetricFamily(
            name="test_dup_metric_family_diff_description",
            description="first description",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        metric_family2 = pb_utils.MetricFamily(
            name="test_dup_metric_family_diff_description",
            description="second description",
            kind=pb_utils.MetricFamily.COUNTER,
        )

        metric2 = metric_family2.Metric()
        self.assertEqual(metric2.value(), 0)

        # Delete metric_family1 and check if metric_family2 still references it
        del metric_family1
        pattern = "test_dup_metric_family_diff_description first description"
        metrics = self._get_metrics()
        self.assertIn(pattern, metrics)

        # The first description will be kept if adding a duplicate metric
        # family name with a different description
        pattern = "test_dup_metric_family_diff_description second description"
        self.assertNotIn(pattern, metrics)

    def test_dup_metric_family(self):
        # Test that adding a duplicate metric family will reuse the original
        # and not add another entry to registry
        metric_family1 = pb_utils.MetricFamily(
            name="test_dup_metric_family",
            description="dup description",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        metric_family2 = pb_utils.MetricFamily(
            name="test_dup_metric_family",
            description="dup description",
            kind=pb_utils.MetricFamily.COUNTER,
        )

        metric_key = "custom_metric_key"
        metric1 = metric_family1.Metric(labels={metric_key: "label1"})
        metric2 = metric_family2.Metric(labels={metric_key: "label2"})

        self.assertEqual(metric1.value(), 0)
        self.assertEqual(metric2.value(), 0)

        patterns = [
            "# HELP test_dup_metric_family dup description",
            "# TYPE test_dup_metric_family counter",
            'test_dup_metric_family{custom_metric_key="label2"} 0',
            'test_dup_metric_family{custom_metric_key="label1"} 0',
        ]
        metrics = self._get_metrics()
        for pattern in patterns:
            self.assertIn(pattern, metrics)

    def test_dup_metric_labels(self):
        # Test that adding a duplicate metric will refer to the same
        # underlying metric, and all instances will be updated
        labels = {"example1": "label1", "example2": "label2"}
        self._dup_metric_helper(labels)

    def test_dup_metric_empty_labels(self):
        # Test that adding a duplicate metric will refer to the same
        # underlying metric, and all instances will be updated
        self._dup_metric_helper()

    def test_metric_lifetime_error(self):
        # Test the error handling when the corresponding 'MetricFamily' is
        # deleted before the 'Metric' is deleted, and the 'Metric' is still
        # being used for metric operations
        kinds = [pb_utils.MetricFamily.COUNTER, pb_utils.MetricFamily.GAUGE]
        metric_family_names = [
            "test_metric_lifetime_error_counter",
            "test_metric_lifetime_error_gauge",
        ]
        for kind, name in zip(kinds, metric_family_names):
            metric_family = pb_utils.MetricFamily(
                name=name, description="test metric lifetime error", kind=kind
            )
            labels = {"example1": "counter_label1", "example2": "counter_label2"}
            metric = metric_family.Metric(labels=labels)

            # Intentionally delete the 'MetricFamily' before the 'Metric' being deleted
            del metric_family

            error_msg = "Invalid metric operation as the corresponding 'MetricFamily' has been deleted."

            # Counter does not support set
            if kind is not pb_utils.MetricFamily.COUNTER:
                with self.assertRaises(pb_utils.TritonModelException) as ex:
                    metric.set(10)
                self.assertIn(error_msg, str(ex.exception))

            with self.assertRaises(pb_utils.TritonModelException) as ex:
                metric.increment(10)
            self.assertIn(error_msg, str(ex.exception))

            with self.assertRaises(pb_utils.TritonModelException) as ex:
                metric.value()
            self.assertIn(error_msg, str(ex.exception))


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for _ in requests:
            # Run the unittest and store the results in InferenceResponse.
            test = unittest.main("model", exit=False)
            responses.append(
                pb_utils.InferenceResponse(
                    [
                        pb_utils.Tensor(
                            "OUTPUT0",
                            np.array([test.result.wasSuccessful()], dtype=np.float16),
                        )
                    ]
                )
            )
        return responses
