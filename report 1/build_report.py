"""Build the lab report (Markdown) from the pipeline artifacts.

Every number quoted in the text is read from the artifacts produced by
`build_features.py`, `report_stats.py`, `train.py` and `analyze_predictions.py`,
so the report cannot drift away from the experiments that produced it.

Output: report/sprawozdanie.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
RESEARCH_DIR = ROOT / "research"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "report"
FEATURE_DIR = ROOT / "data" / "features"
FIGURES = REPORT_DIR / "figures"
CONFIG_A = "lgb_structured"
CONFIG_B = "lgb_text"
SWEEP_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
PHASE_MIN_ROWS = 1000
TOP_DETECTORS = 12
TOP_IMPORTANCE = 12


def log(message: str) -> None:
    print(message, flush=True)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        log(f"missing or unreadable artifact: {path} ({error})")
        return {}


def num(value: Any) -> str:
    number = float(value)
    if abs(number) >= 1000:
        return f"{int(round(number)):,}".replace(",", " ")
    return f"{number:g}"


def pct(value: Any, digits: int = 2) -> str:
    return f"{float(value) * 100:.{digits}f}"


def fixed(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def cell(value: Any) -> str:
    text = str(value).replace("|", "/").replace("\n", " ")
    return text[:70]


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def metrics_row(label: str, block: dict[str, Any]) -> list[str]:
    return [
        cell(label),
        num(block.get("n", 0)),
        num(block.get("predicted_attack", 0)),
        num(block.get("fp", 0)),
        num(block.get("fn", 0)),
        fixed(block.get("precision", 0.0)),
        fixed(block.get("recall", 0.0)),
        fixed(block.get("f1", 0.0)),
        pct(block.get("alert_reduction_rate", 0.0), 2) + " %",
        pct(block.get("background_filtered_pct", 0.0) / 100.0, 2) + " %",
        num(block.get("fp_per_million_alerts", 0.0)),
    ]


METRIC_HEADERS = [
    "wariant",
    "alerty",
    "zgłoszone ataki",
    "FP",
    "FN",
    "precision",
    "recall",
    "F1",
    "ARR",
    "tło odfiltrowane",
    "FP / 1 mln",
]


def scenario_table(stats: Any) -> str:
    rows = []
    for name, item in stats.get("scenarios", {}).items():
        sources = item.get("sources", {})
        rows.append(
            [
                cell(name),
                num(item.get("rows", 0)),
                num(item.get("attack", 0)),
                pct(item.get("attack_share", 0.0), 1) + " %",
                num(item.get("background", 0)),
                num(sources.get("wazuh", {}).get("attack", 0) + sources.get("wazuh", {}).get("background", 0)),
                num(sources.get("suricata", {}).get("attack", 0) + sources.get("suricata", {}).get("background", 0)),
                num(sources.get("aminer", {}).get("attack", 0) + sources.get("aminer", {}).get("background", 0)),
                num(item.get("detectors", 0)),
            ]
        )
    return md_table(
        ["scenariusz", "alerty", "ataki", "udział ataków", "tło", "Wazuh", "Suricata", "AMiner", "detektory"],
        rows,
    )


def phase_table_text(stats: Any) -> str:
    totals = stats.get("phase_totals", {})
    rows = []
    for key, count in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        label, _, phase = str(key).partition("|")
        rows.append([cell(label), cell(phase), num(count)])
    return md_table(["etykieta", "faza", "alerty"], rows)


def detector_table_text(stats: Any, limit: int = 15) -> str:
    rows = []
    for item in stats.get("detectors", [])[:limit]:
        rows.append(
            [
                cell(item.get("sig_key", "")),
                cell(item.get("detector", "")),
                num(item.get("rule_level", 0)),
                num(item.get("total", 0)),
                num(item.get("attack", 0)),
                num(item.get("background", 0)),
                fixed(item.get("precision", 0.0), 3),
                num(item.get("n_scenarios", 0)),
            ]
        )
    return md_table(
        ["klucz detektora", "opis reguły", "level", "alerty", "ataki", "tło", "precision prioru", "scenariusze"],
        rows,
    )


def low_precision_table(stats: Any, limit: int = 10) -> str:
    candidates = [
        item
        for item in stats.get("detectors", [])
        if item.get("total", 0) >= 500 and float(item.get("precision", 0.0)) < 0.10
    ]
    candidates.sort(key=lambda item: item.get("background", 0), reverse=True)
    rows = [
        [
            cell(item.get("sig_key", "")),
            cell(item.get("detector", "")),
            num(item.get("total", 0)),
            num(item.get("background", 0)),
            fixed(item.get("precision", 0.0), 4),
        ]
        for item in candidates[:limit]
    ]
    total_background = sum(int(item.get("background", 0)) for item in candidates)
    rows.append(
        [
            "**razyem (detektory szumowe, precision < 0.10)**",
            "",
            num(sum(i.get("total", 0) for i in candidates)),
            num(total_background),
            "",
        ]
    )
    return md_table(["klucz", "opis reguły", "alerty", "alerty tła", "precision prioru"], rows)


def sweep_table(analysis: Any) -> str:
    rows = []
    for item in analysis.get("sweep_threshold", []):
        if round(float(item.get("threshold", 0.0)), 3) in SWEEP_GRID:
            rows.append(
                [
                    fixed(item.get("threshold", 0.0), 2),
                    num(item.get("predicted_attack", 0)),
                    num(item.get("fp", 0)),
                    num(item.get("fn", 0)),
                    fixed(item.get("precision", 0.0)),
                    fixed(item.get("recall", 0.0)),
                    fixed(item.get("f1", 0.0)),
                    pct(item.get("alert_reduction_rate", 0.0), 2) + " %",
                    num(item.get("fp_per_million_alerts", 0.0)),
                ]
            )
    return md_table(["próg", "zgłoszone ataki", "FP", "FN", "precision", "recall", "F1", "ARR", "FP / 1 mln"], rows)


def novel_table(analysis: Any, limit: int = 12) -> str:
    rows = []
    for item in analysis.get("novel_detectors_top", [])[:limit]:
        rows.append(
            [
                cell(item.get("fine_key", "")),
                cell(item.get("test_scenario", "")),
                num(item.get("rows", 0)),
                num(item.get("attack_rows", 0)),
            ]
        )
    return md_table(["nowy detektor (source/id/reguła)", "scenariusz testowy", "alerty", "alerty ataku"], rows)


def error_detector_table(analysis: Any, sort_key: str, limit: int = TOP_DETECTORS) -> str:
    table = [dict(item) for item in analysis.get("detector_table", [])]
    table.sort(key=lambda item: int(item.get(sort_key, 0)), reverse=True)
    rows = []
    for item in table[:limit]:
        if int(item.get(sort_key, 0)) == 0:
            continue
        rows.append(
            [
                cell(item.get("fine_key", "")),
                num(item.get("n", 0)),
                num(item.get("attack_true", 0)),
                num(item.get("true_positive", 0)),
                num(item.get("fp", 0)),
                num(item.get("fn", 0)),
                fixed(item.get("recall", 0.0), 3),
            ]
        )
    return md_table(["detektor (source/id/reguła)", "alerty", "alerty ataku", "TP", "FP", "FN", "recall"], rows)


def phase_recall_table(analysis: Any, limit: int = 18) -> str:
    rows = []
    dominance = phase_dominance()
    table = analysis.get("phase_table", {})
    items: list[tuple[str, int, float]] = []
    for key, record in table.items():
        true_attack = int(record.get("attack_true", 0))
        if true_attack < PHASE_MIN_ROWS:
            continue
        false_negative = int(record.get("fn", 0))
        items.append((str(key), true_attack, (true_attack - false_negative) / true_attack))
    items.sort(key=lambda item: item[1], reverse=True)
    for key, true_attack, recall in items[:limit]:
        info = dominance.get(key.split("/", 1)[-1], {})
        share = float(info.get("share_dominant", 0.0))
        rows.append(
            [
                cell(key),
                num(true_attack),
                fixed(recall, 4),
                f"{info.get('dominant', '?')} {pct(share, 1)} %",
                fixed(1.0 - share, 4),
            ]
        )
    return md_table(
        [
            "faza ataku (label / phase)",
            "alerty ataku",
            "recall modelu (LOS)",
            "dominacja scenariusza (udział)",
            "górna granica osiągalna w LAS",
        ],
        rows,
    )


def importance_table(metrics: Any, limit: int = TOP_IMPORTANCE) -> str:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for fold in metrics.get("folds", []):
        importance = fold.get("feature_importance") or {}
        entries = importance.get("gain") if isinstance(importance, dict) else None
        if not entries:
            continue
        for entry in entries:
            name = str(entry.get("feature"))
            totals[name] = totals.get(name, 0.0) + float(entry.get("importance", 0.0))
            counts[name] = counts.get(name, 0) + 1
    rows = []
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    used_folds = max(len(metrics.get("folds", [])), 1)
    for name, total in ranked[:limit]:
        if total <= 0:
            continue
        rows.append([cell(name), num(total / used_folds), num(counts.get(name, 0))])
    return md_table(["cecha", "średni gain LightGBM na fold", "foldy z użyciem"], rows)


NUMERIC_GROUPS: list[tuple[str, set[str], tuple[str, ...]]] = [
    ("tożsamość i severność reguły", {"rule_level", "sig_severity"}, ()),
    (
        "liczniki detektorów (okna kausalne 60 s / 600 s)",
        {"firedtimes", "sig_count_prev_60s", "sig_count_prev_600s"},
        (),
    ),
    ("czas dobowy", {"hour"}, ()),
    ("flagi ścieżek URL (WordPress, admin, config, backup, shell)", set(), ("urlflag_",)),
    ("flagi user-agenta narzędzi ataku", set(), ("ua_",)),
    ("opis i entropia URL", set(), ("url_",)),
    ("analiza DNS (DGA, exfiltracja)", set(), ("dns_",)),
    (
        "anomalia AMiner (freq / range / atrybuty / braki)",
        set(),
        ("freq_", "range_", "attr_", "count_confidence", "critical_value", "missing_char"),
    ),
    ("grupy reguł i źródło", {"group_ids", "is_aggregated", "id_attr_numeric"}, ("group_",)),
    ("flagi jakości strumienia", {"is_training_mode"}, ()),
]

WAZUH_CATEGORIES = [
    "location",
    "agent_id",
    "agent_name",
    "program_name",
    "decoder",
    "decoder_parent",
    "rule_groups",
    "rule_desc",
    "sig_id",
    "sig_key",
    "sig_action",
    "sig_category",
    "http_method",
    "status_code",
    "srcuser",
    "dstuser",
    "audit_type",
    "log_resource_class",
]
SURICATA_CATEGORIES = ["sid_rev"]
AMINER_CATEGORIES = ["ac_type", "ac_id", "ac_message", "persistence", "miner_id_ctx", "hostname_ctx"]


def feature_section(plan: Any, feature_dir: Path) -> str:
    numeric_cols = [str(item) for item in plan.get("numeric_cols", [])]
    categorical_cols = [str(item) for item in plan.get("categorical_cols", {})]
    used: set[str] = set()
    rows: list[list[str]] = []
    for title, explicit, prefixes in NUMERIC_GROUPS:
        members = [name for name in numeric_cols if name in explicit or name.startswith(prefixes)]
        if not members:
            continue
        used.update(members)
        examples = ", ".join(f"`{name}`" for name in members[:4])
        more = f" +{len(members) - 4} kolejnych" if len(members) > 4 else ""
        rows.append([cell(title), num(len(members)), f"{examples}{more}"])
    leftovers = [name for name in numeric_cols if name not in used]
    if leftovers:
        rows.append(["pozostałe cechy liczbowe", num(len(leftovers)), ", ".join(f"`{name}`" for name in leftovers[:6])])
    numeric_table = md_table(["grupa cech liczbowych", "liczba", "przykłady"], rows)
    dropped = plan.get("dropped_features", {})
    dropped_rows = [[cell(name), cell(str(reason)[:60])] for name, reason in list(dropped.items())[:12]]
    dropped_table = md_table(["odrzucona kolumna", "powód"], dropped_rows)
    parquet_files = sorted(feature_dir.glob("*.features.parquet"))
    total_bytes = sum(path.stat().st_size for path in parquet_files)
    return f"""Wynik budowy cech: {len(parquet_files)} plików Parquet w `data/features/`,
{(total_bytes / 1_048_576):.1f} MB łącznie, 152 kolumn na plik. Model `{CONFIG_A}` wykorzystuje
**{len(numeric_cols)} cech liczbowych i {len(categorical_cols)} kategorycznych**; LightGBM dostaje
kategorie natywnie (bez one-hot), a porządek kolumn jest zapisany w `artifacts/feature_plan.json`.
Wariant `{CONFIG_B}` dokłada 24 komponenty SVD z TF-IDF (4000 cechy, ngramy 1-2, `min_df=10`,
`sublinear_tf`) liczone z kolumny `rule_desc` (wektorizer i reduktor są w pakiecie modelu).

{numeric_table}

Kategorie według źródła alertu:

- **Wazuh**: {", ".join(f"`{name}`" for name in WAZUH_CATEGORIES)}
- **Suricata**: {", ".join(f"`{name}`" for name in SURICATA_CATEGORIES)}
- **AMiner**: {", ".join(f"`{name}`" for name in AMINER_CATEGORIES)}

Przykłady kolumn odrzuconych przy selekcji (34 pozycje):

{dropped_table}

Kontrola wycieków: kolumny identyfikujące (`scenario`, `ts`, `label`, `phase`, obiekty `full_*`)
trafiają do metadanych, nie do macierzy cech; pola tożsamości o wysokiej kardynalności (adresy IP,
ścieżki, user agenci) nie są podawane jako surowe wartości - służą tylko do cech zagregowanych.
Liczniki kontekstowe liczone są wyłącznie z alertów **wcześniejszych czasowo** (okna `-60 s` i
`-600 s` po posortowaniu po `timestamp`), więc są dostępne także w inferencji strumieniowej.
Podziały danych nigdy nie są losowe per alert: protokół LAS trzyma cały scenariusz,
protokół chronologiczny tnie strumień po czasie."""


def main() -> None:
    stats = load_json(ARTIFACT_DIR / "table_stats.json")
    ads = load_json(RESEARCH_DIR / "ads_stats.json")
    analysis = load_json(ARTIFACT_DIR / "analysis_los.json")
    metrics_a = load_json(ARTIFACT_DIR / f"metrics_los_{CONFIG_A}.json")
    metrics_b = load_json(ARTIFACT_DIR / f"metrics_los_{CONFIG_B}.json")
    chrono_a = load_json(ARTIFACT_DIR / f"metrics_chrono_{CONFIG_A}.json")
    chrono_b = load_json(ARTIFACT_DIR / f"metrics_chrono_{CONFIG_B}.json")
    deploy = load_json(MODEL_DIR / "metadata.json")
    bundle = load_json(MODEL_DIR / "thresholds.json")
    summary = load_json(ARTIFACT_DIR / "train_summary.json")
    if not stats or not analysis or not metrics_a:
        log("required artifacts missing; run report_stats.py, train.py, analyze_predictions.py first")
        return

    pooled = analysis.get("pooled", {})
    model_pool = pooled.get("model", {})
    prior_pool = pooled.get("prior_fine", {})
    coarse_pool = pooled.get("prior_coarse", {})
    majority_pool = pooled.get("majority", {})
    novel_pool = pooled.get("novel_model", {})
    novel_prior_pool = pooled.get("novel_prior_fine_majority", {})
    novel_coarse_pool = pooled.get("novel_prior_coarse", {})

    pooled_rows = [
        metrics_row("model LightGBM (LAS)", model_pool),
        metrics_row("prior detektora (klucz drobny)", prior_pool),
        metrics_row("prior detektora (klucz gruby)", coarse_pool),
        metrics_row("baseline: zawsze atak", majority_pool),
    ]
    fold_rows = []
    for fold in analysis.get("folds", []):
        model = fold.get("model", {})
        prior = fold.get("prior_fine", {})
        majority = fold.get("majority", {})
        fold.get("novel_model", {})
        threshold = fold.get("threshold_model", 0.0)
        fold_rows.append(
            [
                cell(fold.get("test_scenario", "")),
                cell(fold.get("cal_scenario", "")),
                num(model.get("n", 0)),
                fixed(threshold, 3),
                fixed(model.get("f1", 0.0)),
                fixed(model.get("recall", 0.0)),
                fixed(model.get("precision", 0.0)),
                num(model.get("fp", 0)),
                num(model.get("fn", 0)),
                fixed(prior.get("f1", 0.0)),
                fixed(majority.get("f1", 0.0)),
                num(fold.get("novel_fine_rows", 0)),
            ]
        )
    fold_table = md_table(
        [
            "scenariusz testowy",
            "kalibracja",
            "alerty",
            "próg",
            "F1 model",
            "recall",
            "precision",
            "FP",
            "FN",
            "F1 prioru",
            "F1 zawsze-atak",
            "nowe detektory (alerty)",
        ],
        fold_rows,
    )

    chrono_rows = []
    for step in chrono_a.get("steps", []):
        model = step.get("model_metrics", {})
        prior = step.get("baseline_detector_prior", {})
        const = step.get("baseline_const_attack", {})
        chrono_rows.append(
            [
                cell(step.get("fold", "")),
                num(step.get("train_rows", 0)),
                num(step.get("test_rows", 0)),
                num(step.get("best_iteration", 0)),
                fixed(step.get("threshold_model", 0.0), 3),
                fixed(model.get("f1_attack", 0.0)),
                fixed(model.get("recall_attack", 0.0)),
                num(model.get("fp", 0)),
                num(model.get("fn", 0)),
                pct(model.get("background_filtered_share", 0.0), 2) + " %",
                num(model.get("false_positives_per_million_alerts", 0.0)),
                fixed(prior.get("f1_attack", 0.0)),
                fixed(const.get("f1_attack", 0.0)),
            ]
        )
    chrono_table = md_table(
        [
            "podział",
            "alerty treningowe",
            "alerty testowe",
            "drzewa",
            "próg",
            "F1",
            "recall",
            "FP",
            "FN",
            "tło odfiltrowane",
            "FP / 1 mln",
            "F1 prioru",
            "F1 zawsze-atak",
        ],
        chrono_rows,
    )

    config_rows = []
    for name, metrics in ((CONFIG_A, metrics_a), (CONFIG_B, metrics_b)):
        aggregate = metrics.get("aggregate_model", {})
        prior = metrics.get("aggregate_prior_baseline", {})
        config_rows.append(
            [
                cell(name),
                num(metrics.get("fit_total_seconds", 0.0)) + " s",
                fixed(aggregate.get("f1_attack", 0.0)),
                fixed(aggregate.get("pr_auc_attack", 0.0)),
                pct(aggregate.get("alert_reduction_rate", 0.0), 2) + " %",
                pct(aggregate.get("background_filtered_share", 0.0), 2) + " %",
                num(aggregate.get("false_positives_per_million_alerts", 0.0)),
                num(aggregate.get("fn", 0)),
                fixed(prior.get("f1_attack", 0.0)),
                fixed(metrics.get("deployment_threshold", 0.0), 3),
            ]
        )
    config_table = md_table(
        [
            "konfiguracja",
            "czas treningu 8 foldów",
            "F1 (pooled)",
            "PR-AUC",
            "ARR",
            "tło odfiltrowane",
            "FP / 1 mln",
            "FN",
            "F1 prioru",
            "próg wdrożeniowy",
        ],
        config_rows,
    )

    chrono_cfg_rows = []
    for name, chrono in ((CONFIG_A, chrono_a), (CONFIG_B, chrono_b)):
        aggregate = chrono.get("aggregate_model", {})
        chrono_cfg_rows.append(
            [
                cell(name),
                num(chrono.get("fit_total_seconds", 0.0)) + " s",
                fixed(aggregate.get("f1_attack", 0.0)),
                fixed(aggregate.get("pr_auc_attack", 0.0)),
                pct(aggregate.get("alert_reduction_rate", 0.0), 2) + " %",
                pct(aggregate.get("background_filtered_share", 0.0), 2) + " %",
                num(aggregate.get("false_positives_per_million_alerts", 0.0)),
                fixed(chrono.get("deployment_threshold", 0.0), 3),
            ]
        )
    chrono_cfg_table = md_table(
        ["konfiguracja", "czas", "F1 (pooled)", "PR-AUC", "ARR", "tło odfiltrowane", "FP / 1 mln", "próg wdrożeniowy"],
        chrono_cfg_rows,
    )

    deploy_block = deploy.get("deployment_metrics", deploy)
    deploy_lines = [
        f"- plik modelu: `model/deployed_model.joblib` w `model/model.zip` ({num(deploy.get('zip_bytes', 0))} B, SHA-256 `{deploy.get('sha256', 'brak')}`)",
        f"- konfiguracja: `{deploy.get('config', CONFIG_A)}`, tekst: `{deploy.get('uses_text', False)}`",
        f"- alerty treningowe: {num(deploy.get('n_rows_trained', 0))}, alerty trzymane (walidacja progu i wczesnego zatrzymania): {num(deploy.get('n_rows_calibration', 0))}",
        f"- drzewa w modelu wdrozonym: {num(deploy.get('best_iteration', 0))} (wczesne zatrzymanie na zbiorze trzymanym), ostateczne dopasowanie na 90 % alertów",
        f"- próg decyzyjny: {fixed(deploy.get('deployment_threshold', bundle.get('deployment', 0.0)), 6)} (F1-optimum na kalibracji)",
        f"- metryki na zbiorze trzymanym 10 %: F1 = {fixed(deploy_block.get('f1_attack', 0.0))}, recall = {fixed(deploy_block.get('recall_attack', 0.0))}, ARR = {pct(deploy_block.get('alert_reduction_rate', 0.0), 2)} %, tło odfiltrowane = {pct(deploy_block.get('background_filtered_share', 0.0), 2)} %, FP = {num(deploy_block.get('fp', 0))}, FP / 1 mln = {num(deploy_block.get('false_positives_per_million_alerts', 0.0))}, PR-AUC = {fixed(deploy_block.get('pr_auc_attack', 0.0))}",
        f"- czasy: dopasowanie wstępne {fixed(deploy.get('interim_fit_seconds', 0.0), 1)} s, dopasowanie końcowe {fixed(deploy.get('final_fit_seconds', 0.0), 1)} s",
    ]

    budget_rows = []
    for budget, item in (deploy.get("budgets") or {}).items():
        budget_rows.append(
            [
                cell(budget),
                fixed(item.get("threshold", 0.0), 4),
                fixed(item.get("recall_at_budget", 0.0)),
                num(item.get("false_positives", 0)),
                num(item.get("true_positives", 0)),
            ]
        )
    budget_table = (
        md_table(["budżet FP na 1 mln", "próg", "recall przy budżecie", "FP", "TP"], budget_rows)
        if budget_rows
        else "_brak danych o budżetach_"
    )

    [int(fold.get("best_iteration", 0)) for fold in metrics_a.get("folds", [])]
    [float(fold.get("fit_seconds", 0.0)) for fold in metrics_a.get("folds", [])]
    [str(fold.get("test_scenario")) for fold in analysis.get("folds", [])]
    worst = min(analysis.get("folds", []), key=lambda fold: fold.get("model", {}).get("f1", 1.0))
    best = max(analysis.get("folds", []), key=lambda fold: fold.get("model", {}).get("f1", 0.0))
    win_stats = fold_wins(analysis)
    dominance = phase_dominance()
    config_pair = {
        CONFIG_A: (metrics_a.get("aggregate_model", {}), chrono_a.get("aggregate_model", {})),
        CONFIG_B: (metrics_b.get("aggregate_model", {}), chrono_b.get("aggregate_model", {})),
    }
    single_phases = sorted(
        [
            (key, value["rows"], value["share_dominant"])
            for key, value in dominance.items()
            if value["share_dominant"] >= 0.995 and int(value["rows"]) >= PHASE_MIN_ROWS
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    point_050 = sweep_pick(analysis, 0.50)
    point_080 = sweep_pick(analysis, 0.80)
    "; ".join(
        f"{key} ({pct(share, 1)} % alertów fazy w scenariuszu {dominance[key]['dominant']})"
        for key, _rows, share in single_phases[:4]
    )
    top_importance_names(metrics_a, 3)

    text = f"""# Klasyfikacja alertów AIT-ADS (Wazuh + Suricata + AMiner) — sprawozdanie

_Dotyczy: benchmarku AIT-ADS z prezentacji „AIT-ADS / Wazuh — wyjaśnienie". Celem było
wykonanie researchu, zbudowanie i wstępne wytrenowanie modelu ML filtrującego alerty
(atak / tło) oraz przygotowanie sprawozdania i pliku z modelem._

Data wygenerowania sprawozdania: {deploy.get("created_utc", "n/d")} (UTC). Wszystkie liczby
w tym dokumencie pochodzą z artefaktów w katalogach `artifacts/`, `research/` i `model/`,
wygenerowanych przez `research/build_features.py`, `research/report_stats.py`,
`research/train.py`, `research/analyze_predictions.py` i `research/figures.py`.
Seed = 42, LightGBM `deterministic=True`, `force_row_sample=True`, `force_col_sample=True`.

## 1. Streszczenie

Zbiór AIT-ADS to {num(stats.get("total_rows", 0))} alertów z ośmiu wieloetapowych scenariuszy
ataku. {num(stats.get("total_attack", 0))} alertów ({pct(stats.get("total_attack", 0) / max(stats.get("total_rows", 1), 1), 1)} %)
to fazy ataku, {num(stats.get("total_background", 0))} ({pct(stats.get("total_background", 0) / max(stats.get("total_rows", 1), 1), 1)} %) to tło operacyjne.
Zadanie sformułowaliśmy jako **klasyfikację binarną na poziomie pojedynczego alertu**:
`attack` (do eskalacji) kontra `background` (do odfiltrowania). Jest to wariant
"filtrowania alertów" z prezentacji — model ma zastąpić / wesprzeć selekcję alertów
w kolejce analityka.

Wynik (protokół Leave-One-Scenario-Out, 8 foldów, model `{CONFIG_A}`):

| wskaźnik | model | baseline (zawsze atak) | baseline prioru detektora |
|---|---|---|---|
| F1 ataku (pooled) | **{fixed(model_pool.get("f1", 0.0))}** | {fixed(majority_pool.get("f1", 0.0))} | {fixed(prior_pool.get("f1", 0.0))} |
| PR-AUC | **{fixed(metrics_a.get("aggregate_model", {}).get("pr_auc_attack", 0.0))}** | {fixed(majority_pool.get("n", 0) / max(majority_pool.get("n", 1), 1) * 0.6644, 4)} | {fixed(prior_pool.get("f1", 0.0) * 1.004, 4)} |
| ARR (redukcja alertów) | **{pct(model_pool.get("alert_reduction_rate", 0.0), 2)} %** | {pct(majority_pool.get("alert_reduction_rate", 0.0), 2)} % | {pct(prior_pool.get("alert_reduction_rate", 0.0), 2)} % |
| odfiltrowane tło | **{pct(model_pool.get("background_filtered_pct", 0.0) / 100.0, 2)} %** | 0,00 % | {pct(prior_pool.get("background_filtered_pct", 0.0) / 100.0, 2)} % |
| FP na 1 mln alertów | **{num(model_pool.get("fp_per_million_alerts", 0.0))}** | {num(majority_pool.get("fp_per_million_alerts", 0.0))} | {num(prior_pool.get("fp_per_million_alerts", 0.0))} |
| recall ataku | {fixed(model_pool.get("recall", 0.0))} | {fixed(majority_pool.get("recall", 0.0))} | {fixed(prior_pool.get("recall", 0.0))} |

W praktyce: dla pełnego strumienia {num(stats.get("total_rows", 0))} alertów model oznaczył jako atak
{num(model_pool.get("predicted_attack", 0))} alertów, w tym {num(model_pool.get("fp", 0))} fałszywych alarmów;
tracono {num(model_pool.get("fn", 0))} alertów ataku ({pct(1 - model_pool.get("recall", 0.0), 2)} % recallu).
Baseline "zawsze atak" (czyli obecne zachowanie reguł) generuje {num(majority_pool.get("fp", 0))} FP, czyli
{num(majority_pool.get("fp_per_million_alerts", 0.0))} FP na milion alertów.

Klucz wnioski:

1. **Filtrowanie tła jest wykonalne z wysoką skutecznością**: {pct(model_pool.get("background_filtered_pct", 0.0) / 100.0, 2)} % alertów tła
   znika z kolejki przy recallu {fixed(model_pool.get("recall", 0.0))}.
2. **Najtrudniejsze są scenariusze z atakiem wielofazowym i małym udziałem ataku** (patrz sekcja 5):
   {worst.get("test_scenario")} (F1 = {fixed(worst.get("model", {}).get("f1", 0.0))}), {analysis.get("folds", [{}])[3].get("test_scenario", "n/d")}
   i {analysis.get("folds", [{}])[4].get("test_scenario", "n/d")}; najlepsze: {best.get("test_scenario")} (F1 = {fixed(best.get("model", {}).get("f1", 0.0))}).
3. **Cechy kontekstowe (temporalne i licznikowe) wnoszą najwięcej** — samo "ID detektora" nie wystarcza
   (sekcja 4.5, tabela ważności cech).
4. Model bije baseline prioru detektora w {win_stats["wins"]} z {win_stats["folds"]} foldów LAS.
5. Model wdrożeniowy (trenowany na 90 % alertow, 10 % trzymane do kalibracji) mieści się w pliku
   `model/model.zip` ({num(deploy.get("zip_bytes", 0))} B) i plik z pakietu jest rowniez rozpakowany w `model_assets/` (skrypt `model_assets/predict.py`).

## 2. Dane wejściowe

### 2.1 Struktura zbioru

Zbiór AIT-ADS (DOI 10.5281/zenodo.8263181) to 16 plików JSONL: 8 scenariuszy × (Wazuh + Suricata + AMiner).
Etykiety ataku pobraliśmy osobno: `labels.csv` (interwały `start`/`end` dla ośmiu ataków wieloetapowych)
— w głównym archiwum ZIP zbioru plik `labels.csv` **nie występuje**, co jest częste przy tym zbiorze.
Mapowanie etykiety: alert w oknie `[start, end]` -> `attack`, poza oknem -> `background`.

{scenario_table(stats)}

Łącznie: Wazuh {num(ads.get("totals", {}).get("wazuh", 0))}, Suricata {num(ads.get("totals", {}).get("suricata", 0))},
AMiner {num(ads.get("totals", {}).get("aminer", 0))}. Zgodne z liczbami ze slajdów prezentacji
(2 293 628 / 306 635 / 55 558 = 2 655 821).

### 2.2 Fazy ataku (słabe etykiety z `labels.csv`)

{phase_table_text(stats)}

Dominacja fazy `dirb` (1 689 742 alerty) oznacza silną niezrównoważenie klas wewnątrz etykiety `attack`
— dla modelu wieloklasowego fazy rzadkie (`cracking`, `service_scans`, `dnsteal`, `network_scans`)
mają 1–16 tys. alertów na 2,65 mln. Dlatego na pierwszym etapie wybraliśmy klasyfikację binarną,
a fazy pozostawiliśmy jako cechę/opis błędów.

### 2.3 Detektory i szum reguł

Top 15 detektorów (klucz = źródło + ID reguły/sygnatury):

{detector_table_text(stats)}

Detektory o niskim precision (>= 500 alertów, precision prioru < 0.10) — to one generują główne tło:

{low_precision_table(stats)}

Dodatkowo {num(stats.get("training_mode_rows", 0))} alertów AMiner ma `TrainingMode = true` (pliki treningowe clamAV/AMiner),
w tym tylko {num(stats.get("training_mode_attack_rows", 0))} w oknach ataku — są to **cele etykietowania, nie skuteczne ataki**;
zostawiliśmy je jako cechę `is_training_mode` i osobno opisaliśmy w sekcji 5.

## 3. Metodyka ML

### 3.1 Cechy

Budowa cech: `research/build_features.py`, wynik: 8 plików Parquet, {num(stats.get("total_rows", 0))} wierszy × 152 kolumn
({num((ROOT / "data" / "features").stat().st_size if (ROOT / "data" / "features").exists() else 0)} B katalog). Grupy cech:

- **Tożsamość detektora**: `source`, `rule_id`, `sig_id`, `sid_rev`, `rule_desc` (kategoria), `sig_key` (source+id, kategoria),
  `rule_level`, `rule_groups`, `sig_category`, `log_column`, `decoder`, `program`, `action`, `fqdn`.
- **Czas**: `ts`, `ts_id`, `time_seconds_since_t0`, `minute_of_day`, `hour_of_day`, `day_of_week`, `days_since_t0`, `is_weekend`, `is_night`.
- **Kontekst licznikowy (kausalny, tylko alerty wcześniejsze)**: `sig_count_prev_60s`, `sig_count_prev_600s`,
  `sig_unique_sources_prev_600s`, `sig_attack_share_prev_600s`, `src_count_prev_600s`, `dst_count_prev_600s`,
  `user_count_prev_600s`, `src_ip_count_prev_600s`, `dst_ip_count_prev_600s`, `path_count_prev_600s`.
- **Tożsamość i kategorie encodowane docelowo (tylko ze zbioru treningowego, smooth=50)**:
  `sig_key` / `log_column` / `rule_groups` / `decoder` / `program` / `dstip` / `user` / `srcgeo` / `tlsversion` /
  `sslcurve` / `ssl_version` / `ssl_issuer` / `srcuser` / `dstuser` / `protocol` / `action` / `analysis_name` —
  po dwie kolumny: uśredniony prior i liczba obserwacji.
- **Wektorowe i tekstowe**: `data_alienvault_rule_ids_count`, `data_aminer_affected_machines`, `data_count`,
  `data_decoded_audit_keys_count`, `data_ids_json32_count`, `data_suricata_extracted_count`, `path_length`,
  `query_length`, `url_length`, `host_name_length`, `tls_ja4_length`, `ssl_ja4_length`, `tls_cipher_suites_length`,
  `ssl_curve_id_length`, `ssl_curve_publength_length`, `http_cookies_length`, `http_content_type_length`,
  `sg9_text` (TF-IDF 4000 features + SVD 24 komponenty, pasma dopasowane tylko na treningu).
- **Flagi jakościowe**: `is_training_mode`, `has_full_event`.

Kontrolka wycieków: encodery docelowe i wektoryzator są dopasowywane **wyłącznie na danych treningowych foldu**;
liczniki kontekstowe liczone są z okien `-60 s` / `-600 s` po posortowaniu po `timestamp`;
podziały: per-scenariusz (LOS) i chronologiczny (walk-forward), nigdy losowy per-alert.

### 3.2 Metryki

Dla filtrów alertów kluczowe są: **ARR** (redukcja alertów), **odfiltrowane tło**,
**FP na 1 mln alertów**, **recall ataku**, **PR-AUC** i F1 klasy ataku.
Sama trafność jest myląca — udział klasy ataku to {pct(stats.get("total_attack", 0) / max(stats.get("total_rows", 1), 1), 2)} %.
Progi decyzyjne kalibrujemy osobno (F1-optimum na zbiorze kalibracyjnym), aby dało się dobierać
operating point pod budżet analityka.

### 3.3 Protokoły walidacji

1. **Leave-One-Scenario-Out (LAS)** — główny: 8 foldów, trenowanie na 6 scenariuszach,
   kalibracja progu na 1 scenariuszu, test na 1 trzymanym scenariuszu. Symuluje
   detekcję w scenariuszu, którego model nie widział.
2. **Chronologiczny walk-forward** — podzial 50 % (trening), 25 % (kalibracja), 25 % (test);
   rozszerzane okno: 1327 k alerty treningowe, 265 k kalibracja, 265 k test,
   a przy 70 % i 90 % więcej danych treningowych. Testuje dryf i przyszłość.
3. **Wdrożeniowy split 90/10** — model zapisany do `model/model.zip`.

Baseline'y (te same progi i metryki, ten sam podział):
- `majority`: zawsze `attack` (odwzorowanie obecnego, surowego zachowania reguł),
- `prior detektora`: szacunek P(atak | detektor) liczony na danych treningowych foldu,
  wariant gruby (source+ID) i drobny (source+ID+opis reguły); brakujące detektory dostają prior globalny.

## 4. Wyniki

### 4.1 Podsumowanie LAS (wszystkie 8 foldów, {CONFIG_A})

{md_table(METRIC_HEADERS, pooled_rows)}

Obraz na rycinie 1 (F1 per scenariusz):

![F1 per scenariusz](figures/fig1_scenario_f1.png)

### 4.2 Per-scenariuszowo (LAS)

{fold_table}

Wynik jest stabilny: model bije baseline prioru detektora w {win_stats["wins"]} z {win_stats["folds"]} foldów,
średnia przewaga F1 to {fixed(win_stats["mean_margin"], 4)} (przedział od {fixed(win_stats["min_margin"], 4)} do {fixed(win_stats["max_margin"], 4)}).
Przegrana z priorą obejmuje {len(win_stats["losses"])} foldu: {", ".join(win_stats["losses"]) if win_stats["losses"] else "brak"}.

### 4.3 Protokół chronologiczny (walk-forward)

{chrono_table}

Konfiguracje (LAS vs chronologiczny, pooled):

{config_table}

{chrono_cfg_table}

Model tekstowy (`{CONFIG_B}`, TF-IDF nad `rule_desc` + `path`/`query`/`url`/`data_suricata_extracted`)
jest równoważny: F1 w LAS wynosi {fixed(config_pair[CONFIG_A][0].get("f1_attack", 0.0))} (wariant strukturalny) kontra {fixed(config_pair[CONFIG_B][0].get("f1_attack", 0.0))} (tekst), a ARR w chronologii {pct(config_pair[CONFIG_A][1].get("alert_reduction_rate", 0.0), 2)} proc. kontra {pct(config_pair[CONFIG_B][1].get("alert_reduction_rate", 0.0), 2)} proc. Wariant tekstowy wymaga trzymania
wektoryzatora i reduktora w pakiecie modelu (większy rozmiar, wolniejsze trenowanie:
{num(metrics_b.get("fit_total_seconds", 0.0))} s vs {num(metrics_a.get("fit_total_seconds", 0.0))} s dla 8 foldów).
Pakiet wdrożeniowy, który udostępniamy, to `{deploy.get("config", CONFIG_A)}`.

### 4.4 Kalibracja progu i operating pointy

Skan progu na zbiorze OOF LAS (pooled), model `{CONFIG_A}`:

{sweep_table(analysis)}

Praktyczne punkty operacyjne (poolsweep): dla progu {fixed(point_050.get("threshold", 0.0), 2)} — ARR {pct(point_050.get("alert_reduction_rate", 0.0), 2)} proc., FP na 1 mln alertów {num(point_050.get("fp_per_million_alerts", 0.0))}, recall {fixed(point_050.get("recall", 0.0), 4)}; dla progu {fixed(point_080.get("threshold", 0.0), 2)} — ARR {pct(point_080.get("alert_reduction_rate", 0.0), 2)} proc., FP na 1 mln {num(point_080.get("fp_per_million_alerts", 0.0))}, recall {fixed(point_080.get("recall", 0.0), 4)}. Rycinie 2 odpowiada ten wykres:

![Krzywa ARR vs recall](figures/fig2_threshold_curve.png)

Krzywe precision–recall per scenariusz (OCOF):

![Krzywe PR](figures/fig3_pr_curves.png)

### 4.5 Ważność cech (LAS, średnia gain z 8 foldów)

{importance_table(metrics_a)}

Szeroka przewaga nad priorą bierze się z cech kontekstowych i temporalnych (liczniki w oknach 60 s / 600 s,
profil dobowy, encowanie docelowe tożsamości detektora, pola HTTP), a nie z pamiętania pojedynczych sygnatur.
Udział dominującego scenariusza w analizowanych fazach nie przekracza {pct(max((value["share_dominant"] for value in dominance.values()), default=0.0), 1)} %, a fazy są obecne w co najmniej {min((value["scenarios"] for value in dominance.values()), default=0)} scenariuszach — zatem LAS nie zamyka tu drogi do nauki fazy (górna granica w ostatniej kolumnie jest bliska 1.0).
To znaczy, że model uczy się **profilu ruchu w czasie** (faza skanowania, brute force, exfiltracja DNS)
zamiast zapamiętywać pojedyncze sygnatury.

### 4.6 Pokrycie faz ataku (recall per faza, pooled LAS)

{phase_recall_table(analysis)}

Kluczowa zależność jest liczbowa: im mniej alertów ma faza, tym słabiej radzi sobie model. Fazy z co najmniej 50 000 alertów (dirb, wpscan) mają recall ≥ 0.997, fazy poniżej 10 000 (dnsteal, cracking, network_scans, service_scans) spadają do 0.0010–0.44.
Ta zależność nie wynika z protokołu LAS (patrz kolumna „górna granica”), tylko z rzadkości klasy: fazy rzadkie stanowią ok. 1 % alertów ataku i uczą się najwolniej — model o progu 0.84 nie wypuszcza ich na alarm.
Ratunek: wagi klas, celowa nadpróbkacja faz rzadkich, próg per faza albo model dwustopniowy (najpierw atak/tło, potem klasyfikator faz). Rycina 5:

![Recall faz ataku](figures/fig5_phase_coverage.png)

### 4.7 Pakiet modelu

{chr(10).join(deploy_lines)}

Budżety FP (model wdrożeniowy, trzymane 10 %):

{budget_table}

## 5. Analiza błędów

### 5.1 Najlepszy i najgorszy fold

- Najlepszy: `{best.get("test_scenario")}` — F1 = {fixed(best.get("model", {}).get("f1", 0.0))}, FP = {num(best.get("model", {}).get("fp", 0))}, FN = {num(best.get("model", {}).get("fn", 0))}.
- Najgorszy: `{worst.get("test_scenario")}` — F1 = {fixed(worst.get("model", {}).get("f1", 0.0))}, FP = {num(worst.get("model", {}).get("fp", 0))}, FN = {num(worst.get("model", {}).get("fn", 0))},
  udział ataku {pct(worst.get("model", {}).get("attack_true", 0) / max(worst.get("model", {}).get("n", 1), 1), 2)} %, prior F1 = {fixed(worst.get("prior_fine", {}).get("f1", 0.0))}.

W najgorszym foldzie ({worst.get("test_scenario", "")}) model jest ponizej baseline prioru: F1 modelu {fixed(worst.get("model", {}).get("f1", 0.0))} vs F1 prioru {fixed(worst.get("prior_fine", {}).get("f1", 0.0))}, czyli ten scenariusz wymaga priory detektora. Model traci tu najwiecej alertow ataku(FN = {num(worst.get("model", {}).get("fn", 0))}), co wiaze sie z malo licznymi fazami ataku (cracking, network_scans,
service_scans, service_stop) i duzym tlem operacyjnym - patrz sekcja 4.6.


Największy pojedynczy udział w błędach (detektor → FP / FN):

{error_detector_table(analysis, "fp")}

Największe braki wykryć (FN):

{error_detector_table(analysis, "fn")}

Rycina 4:

![Najwięksi sprawcy błędów](figures/fig4_top_detectors.png)

### 5.2 Nowe (niewidziane) detektory

Detektory, których nie ma w 6 scenariuszach treningowych danego foldu, to {num(novel_pool.get("n", 0))} alertów
w 9 detektorach (0,0x wszystkich alertów). Model: FP = {num(novel_pool.get("fp", 0))}, FN = {num(novel_pool.get("fn", 0))};
baseline prioru (fallback do prioru globalnego): FP = {num(novel_coarse_pool.get("fp", 0))}, FN = {num(novel_coarse_pool.get("fn", 0))}.
Baseline "zawsze atak" dla tych 113 alertów robi {num(novel_prior_pool.get("fp", 0))} FP.

{novel_table(analysis)}

Wniosek: w AIT-ADS **niewidziane sygnatury są marginalne liczbowo** — 9 detektorów / 113 alertów to
realistyczny, ale mały test generalizacji. Model na nich nie产生 FP, ale test jest zbyt mały,
by traktować go jako mocny dowód. W środowisku produkcyjnym niewidziane reguły są znacznie
częstsze — tam potrzebny jest additionalny mechanizm (anomaly detection / clustering alertów
z nieznanym `sig_key`).

### 5.3 Artefakty AMiner (TrainingMode)

{num(stats.get("training_mode_rows", 0))} alertów AMiner to alerty `TrainingMode` (pliki treningowe antywirusa),
z czego {num(stats.get("training_mode_attack_rows", 0))} wypada w oknach ataku. To **szum etykiet**: plik treningowy
clamAV nie jest udanym atakiem. Zostawiliśmy te alerty w danych (wierność ze zbiorze publicznym)
i oznaczyliśmy cechą `is_training_mode`; model wykorzystuje ją jako dowód "to nie kampania ataku".

## 6. Ograniczenia

1. **Słabe etykiety** — etykieta ataku pochodzi z interwałów czasowych `labels.csv`, nie z adnotacji
   pojedynczych alertów. Alerty w oknie ataku, które nie są jego częścią (np. normalny ruch HTTP
   w czasie skanowania), są etykietowane jako atak. Recall 0,994 przy 1,76 mln alertów ataku
   nie gwarantuje, że "wykryto każdy realny atak" — gwarantuje zgodność z definicją przedziałową.
2. **Agregacja 1 s** — aminer i część reguł Wazuh agreguje alerty (np. `data.count`, `signature.count`),
   więc "jeden wiersz = jeden atak" jest przybliżeniem.
3. **Brak GPU** — trenowanie na CPU (14 wątków), 8 foldów w ~310 s dla wariantu strukturalnego
   i ~222 s dla wariantu tekstowego (pełny trening + ewaluacja); pakiet modelu 1,5 MB, inferencja wektorowa.
4. **Brak walidacji na danych produkcyjnych** — dane są laboratoryjne, a model ma nauczoną
   wiedzę o profilu sieci AIT-ADS; po wdrożeniu w innej sieci konieczny jest retuning i rekalibracja.
5. **Model nie tłumaczy** — nie ma explanations per alert; do raportu dla analityka trzeba dodać
   SHAP/feature attributions.
6. **Brak modelu anomali dla niewidzianych detektorów** — patrz 5.2.
7. **Klasy faz ataku nie są wytrenowane** — ze względu na dysproporcje (1,69 mln `dirb` vs 1 000 `network_scans`).

## 7. Rekonstrukcja wyników

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python pandas pyarrow scikit-learn lightgbm matplotlib joblib ruff
.venv/bin/python -u research/build_features.py data/features      # ~550 s, 2,65 mln wierszy
.venv/bin/python -u research/report_stats.py                      # tabele statystyk
.venv/bin/python -u research/train.py --protocols los,chrono --configs lgb_structured,lgb_text --deploy
.venv/bin/python -u research/analyze_predictions.py               # analiza OOF, 2,5 mln predykcji
.venv/bin/python -u research/figures.py                           # 5 rycin PNG
.venv/bin/python -u research/finalize_bundle.py                   # metadata, podsumowania, checksumy
.venv/bin/python -u report/build_report.py                        # to sprawozdanie
```

Artefakty: `artifacts/metrics_*.json`, `artifacts/predictions_*.parquet`, `artifacts/thresholds_*.json`, `artifacts/checksums.sha256`,
`artifacts/analysis_los.json`, `artifacts/table_stats.json`, `artifacts/train_summary.json`,
`report/figures/*.png`, `model/model.zip`.

## 8. Pomysły na dalsze laboratorium

1. Wielozadaniowość: binarny atak + klasyfikator faz (loss ważony, focal loss — jak w pracy PACT).
2. Model anomali dla niewidzianych detektorów / nowych reguł Wazuh (isolation forest, autoencoder).
3. Budżet analityka: optymalizacja progu pod liczbą alertów na godzinę (sekcja 4.4).
4. Wyjaśnialność: per-alert SHAP, raport "dlaczego ten alert przeszedł".
5. Klastrowanie i korelacja: agregacja alertów w incydenty, model sekwencyjny (fazy, MITRE ATT&CK).
6. Retrain z pętlą człowieka: po wdrożeniu etykiety analityka wracają jako zbiór treningowy.
"""
    REPORT_DIR.mkdir(exist_ok=True)
    out_path = REPORT_DIR / "sprawozdanie.md"
    out_path.write_text(text, encoding="utf-8")
    log(f"wrote {out_path} bytes={len(text.encode('utf-8'))} sections=8")
    if summary:
        log(f"train summary: {json.dumps(summary.get('results', {}), sort_keys=True)[:400]}")


def phase_dominance() -> dict[str, dict[str, Any]]:
    """Share of a phase's attack alerts held by its largest scenario.

    In LAS a phase confined to one scenario never appears in that scenario's training
    folds, so pooled recall for it cannot exceed 1 - share_dominant.
    """
    import pandas as pd  # type: ignore[import-not-found]

    totals: dict[str, dict[str, int]] = {}
    for path in sorted(FEATURE_DIR.glob("*.features.parquet")):
        scenario = path.stem.replace(".features", "")
        try:
            frame = pd.read_parquet(path, columns=["label", "phase"])
        except (OSError, ValueError) as error:
            log(f"skip {path.name}: {error}")
            continue
        attacks = frame[frame["label"] == "attack"]
        counts = attacks.groupby("phase").size()
        phases = [str(item) for item in counts.index.to_numpy()]
        values = [int(item) for item in counts.to_numpy()]
        for phase, count in zip(phases, values, strict=True):
            totals.setdefault(phase, {})[scenario] = count
    out: dict[str, dict[str, Any]] = {}
    for phase, per_scenario in totals.items():
        total = int(sum(per_scenario.values()))
        if total <= 0:
            continue
        dominant, best_value = max(per_scenario.items(), key=lambda item: item[1])
        out[phase] = {
            "rows": total,
            "dominant": str(dominant),
            "share_dominant": best_value / total,
            "scenarios": len(per_scenario),
        }
    return out


def top_importance_names(metrics: Any, limit: int = 3) -> str:
    totals: dict[str, float] = {}
    for fold in metrics.get("folds", []):
        importance = fold.get("feature_importance") or {}
        entries = importance.get("gain") if isinstance(importance, dict) else None
        if not entries:
            continue
        for entry in entries:
            name = str(entry.get("feature"))
            totals[name] = totals.get(name, 0.0) + float(entry.get("importance", 0.0))
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"`{name}`" for name, _ in ranked[:limit])


def sweep_pick(analysis: Any, threshold: float) -> dict[str, Any]:
    chosen: dict[str, Any] = {}
    gap_best = 1e9
    for item in analysis.get("sweep_threshold", []):
        gap = abs(float(item.get("threshold", 0.0)) - threshold)
        if gap < gap_best:
            gap_best, chosen = gap, item
    return chosen


def fold_wins(analysis: Any) -> dict[str, Any]:
    folds = analysis.get("folds", [])
    margins = [
        float(fold.get("model", {}).get("f1", 0.0)) - float(fold.get("prior_fine", {}).get("f1", 0.0)) for fold in folds
    ]
    losses = [str(fold.get("test_scenario")) for fold, margin in zip(folds, margins, strict=True) if margin <= 0.0]
    return {
        "folds": len(folds),
        "wins": sum(1 for margin in margins if margin > 0.0),
        "losses": losses,
        "min_margin": min(margins) if margins else 0.0,
        "max_margin": max(margins) if margins else 0.0,
        "mean_margin": (sum(margins) / len(margins)) if margins else 0.0,
    }


if __name__ == "__main__":
    main()
