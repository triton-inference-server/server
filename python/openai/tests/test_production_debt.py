# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../openai_frontend/production_debt.py",
)
spec = importlib.util.spec_from_file_location("triton_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["triton_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtServingGate = production_debt_mod.ProductionDebtServingGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtServingGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtServingGate(
            never_equate_intent_to_approval=True,
            max_acceptable_tdi=12.0,
        )

    def test_clean_serving_passes_readiness(self) -> None:
        report = self.gate.evaluate_serving_pipeline(
            server_id="triton_k8s_cluster_node_01",
            configured_max_queue_ms=50,
            actual_queue_delay_ms=52,
            ensemble_latency_seconds=0.35,
            model_reload_failures=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.tdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_serving_fails_debt(self) -> None:
        report = self.gate.evaluate_serving_pipeline(
            server_id="uncalibrated_triton_ensemble",
            configured_max_queue_ms=50,
            actual_queue_delay_ms=160,  # High queue sprawl (3.2x)
            ensemble_latency_seconds=2.8,  # High latency
            model_reload_failures=3,  # 3 hot reload failures
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.tdi_score, 50.0)
        self.assertIn("HIGH_DYNAMIC_BATCHING_QUEUE_SPRAWL_3.20X", report.critical_smells)
        self.assertIn("HIGH_ENSEMBLE_LATENCY_2.80S", report.critical_smells)
        self.assertIn("DETECTED_3_MODEL_REPOSITORY_HOT_RELOAD_FAILURES", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_SERVING_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_serving_pipeline("server-1")
        self.gate.evaluate_serving_pipeline("server-2")
        self.gate.evaluate_serving_pipeline("server-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
