
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class MulticlassGRU(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=32):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.classifier(output[:, -1, :])


class WazuhPredictor:
    def __init__(self, model_path):
        self.model_path = Path(model_path)

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=True
        )

        self.phase_names = list(checkpoint["phase_names"])
        self.feature_names = list(checkpoint["feature_names"])
        self.seq_len = int(checkpoint["sequence_length"])
        self.window_seconds = int(checkpoint["window_seconds"])

        self.model = MulticlassGRU(
            input_size=len(self.feature_names),
            num_classes=len(self.phase_names),
            hidden_size=int(checkpoint["hidden_size"])
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.eval()

        self.rule_features = [
            name for name in self.feature_names
            if name.startswith("rule_count_")
        ]

    def build_sequence(self, events, scenario, target_start):
        """Odtwarza sekwencję wejściową GRU z alertów."""

        target = pd.Timestamp(target_start)

        if target.tzinfo is None:
            target = target.tz_localize("UTC")
        else:
            target = target.tz_convert("UTC")

        history_start = target - pd.Timedelta(
            seconds=(self.seq_len - 1) * self.window_seconds
        )
        history_end = target + pd.Timedelta(
            seconds=self.window_seconds
        )

        subset = events.loc[
            events["scenario"].eq(scenario)
            & (
                pd.to_datetime(events["event_time"], utc=True)
                >= history_start
            )
            & (
                pd.to_datetime(events["event_time"], utc=True)
                < history_end
            ),
            ["event_time", "rule_id", "rule_level"]
        ].copy()

        subset["event_time"] = pd.to_datetime(
            subset["event_time"], utc=True
        )

        subset["rule_id"] = subset["rule_id"].astype(str)

        sequence = np.zeros(
            (self.seq_len, len(self.feature_names)),
            dtype=np.float32
        )

        for step in range(self.seq_len):
            start = history_start + pd.Timedelta(
                seconds=step * self.window_seconds
            )
            end = start + pd.Timedelta(
                seconds=self.window_seconds
            )

            part = subset.loc[
                (subset["event_time"] >= start)
                & (subset["event_time"] < end)
            ]

            values = {
                "alert_count": len(part),
                "unique_rules": part["rule_id"].nunique(),
                "max_rule_level": (
                    part["rule_level"].max()
                    if len(part) else 0
                )
            }

            for feature in self.rule_features:
                rule_id = feature.removeprefix("rule_count_")
                values[feature] = int(
                    (part["rule_id"] == rule_id).sum()
                )

            for j, feature in enumerate(self.feature_names):
                value = float(values[feature])

                if feature != "max_rule_level":
                    value = np.log1p(value)

                sequence[step, j] = value

        return sequence

    def predict(self, events, scenario, target_start):
        sequence = self.build_sequence(
            events, scenario, target_start
        )

        # Trenowaliśmy na oknach zawierających alerty.
        # Nie klasyfikujemy pustego okna docelowego.
        if sequence[-1, self.feature_names.index("alert_count")] == 0:
            raise ValueError(
                "Brak alertów w oknie docelowym; "
                "model nie był trenowany dla takich okien."
            )

        x = torch.from_numpy(
            sequence
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            scores = torch.softmax(logits, dim=1)[0].numpy()

        predicted_idx = int(np.argmax(scores))

        return {
            "predicted_phase": self.phase_names[predicted_idx],
            "predicted_score": float(scores[predicted_idx]),
            "scores": {
                phase: float(score)
                for phase, score in zip(self.phase_names, scores)
            }
        }


def wazuh_json_to_dataframe(alerts, scenario="incoming"):
    """Konwersja surowych alertów Wazuh do formatu modelu."""
    rows = []

    for alert in alerts:
        location = str(alert.get("location") or "").lower()

        # Pomijamy alerty pochodzące z Suricaty.
        if "suricata" in location:
            continue

        decoder = alert.get("decoder") or {}
        if decoder.get("name") in {"json", "snort"}:
            continue

        rule = alert.get("rule") or {}
        timestamp = alert.get("@timestamp")

        if not timestamp:
            continue

        if rule.get("id") is None or rule.get("level") is None:
            continue

        try:
            event_time = pd.to_datetime(
                timestamp, utc=True
            )
            rule_level = float(rule["level"])
        except (ValueError, TypeError):
            continue

        rows.append({
            "scenario": scenario,
            "event_time": event_time,
            "rule_id": str(rule["id"]),
            "rule_level": rule_level
        })

    return pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "event_time",
            "rule_id",
            "rule_level"
        ]
    )


def predict_wazuh_json(
    predictor,
    alerts,
    target_start,
    scenario="incoming"
):
    """
    Pełna predykcja z surowych JSON-ów Wazuh.

    alerts: lista alertów obejmująca historię 60 sekund
    target_start: początek analizowanego okna 10 s (UTC)
    """
    events = wazuh_json_to_dataframe(
        alerts,
        scenario=scenario
    )

    return predictor.predict(
        events,
        scenario=scenario,
        target_start=target_start
    )
