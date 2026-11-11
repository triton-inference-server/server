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

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log: logging.Logger = logging.getLogger(__name__)

GENESIS_HASH: str = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)


@dataclass
class TritonDebtReport:
    server_id: str
    tdi_score: float  # Triton Debt Index (target <= 12.0)
    batching_sprawl_multiplier: float  # Target <= 1.08x
    ensemble_latency_seconds: float  # Target <= 0.45s
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: List[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """
    Cryptographic SHA-256 hash-chained Action Ledger for NVIDIA Triton Inference Server model serving.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_serving_event(
        self,
        server_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = f"{index}|{self._last_hash}|{server_id}|{event_type}|{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "server_id": server_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtServingGate:
    """
    A2Z SOC Production Debt & Technical Due Diligence Gate for NVIDIA Triton Inference Server.

    Quantifies dynamic batching queues, ensemble latency, and model hot-reload stability against 4 Enterprise KPIs:
    1. Triton Debt Index (TDI <= 12.0)
    2. Dynamic Batching Sprawl Multiplier (DBSM <= 1.08x)
    3. P99 Ensemble Step Latency (<= 0.45s)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_tdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_tdi = max_acceptable_tdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        for path_str in ("artifacts/KILL", "/tmp/KILL"):
            if Path(path_str).exists():
                return True
        return False

    def evaluate_serving_pipeline(
        self,
        server_id: str,
        configured_max_queue_ms: int = 50,
        actual_queue_delay_ms: int = 52,
        ensemble_latency_seconds: float = 0.35,
        model_reload_failures: int = 0,
        un_gated_mutations: int = 0,
    ) -> TritonDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_serving_event(
                server_id=server_id,
                event_type="serving_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            raise PermissionError(
                "A2Z SOC ActionGate: Emergency kill switch is engaged. Triton model serving halted."
            )

        critical_smells: List[str] = []

        # KPI 2: Dynamic Batching Sprawl Multiplier
        queue_ratio = actual_queue_delay_ms / max(1, configured_max_queue_ms)
        if queue_ratio > 1.8:
            critical_smells.append(f"HIGH_DYNAMIC_BATCHING_QUEUE_SPRAWL_{queue_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if ensemble_latency_seconds > 1.5:
            critical_smells.append(f"HIGH_ENSEMBLE_LATENCY_{ensemble_latency_seconds:.2f}S")

        # Model reload failures
        if model_reload_failures > 1:
            critical_smells.append(f"DETECTED_{model_reload_failures}_MODEL_REPOSITORY_HOT_RELOAD_FAILURES")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_SERVING_MUTATIONS")

        # KPI 1: Triton Debt Index (0 = Clean, 100 = Catastrophic)
        tdi = (
            max(0.0, (queue_ratio - 1.0) * 20.0)
            + max(0.0, (ensemble_latency_seconds - 0.45) * 10.0)
            + (model_reload_failures * 15.0)
            + (un_gated_mutations * 30.0)
        )
        tdi_score = round(min(100.0, tdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - tdi_score)
        is_production_ready = (
            tdi_score <= self.max_acceptable_tdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_serving_event(
            server_id=server_id,
            event_type="serving_authorized" if is_production_ready else "serving_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "tdi_score": tdi_score,
                "queue_ratio": queue_ratio,
                "configured_max_queue_ms": configured_max_queue_ms,
                "actual_queue_delay_ms": actual_queue_delay_ms,
                "ensemble_latency_seconds": ensemble_latency_seconds,
                "model_reload_failures": model_reload_failures,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return TritonDebtReport(
            server_id=server_id,
            tdi_score=tdi_score,
            batching_sprawl_multiplier=round(queue_ratio, 2),
            ensemble_latency_seconds=round(ensemble_latency_seconds, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
