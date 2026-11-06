# Klasyfikacja alertow AIT-ADS (strumien Wazuh)

Repozytorium laboratoryjne: model ML do filtrowania alertow ataku versus tlo na publicznym
zbiorze AIT Alert Data Set (AIT-ADS, DOI 10.5281/zenodo.8263181). Sprawozdanie jest w Markdownie.

**Zakres tej edycji laboratorium:** modelowanie **tylko strumienia Wazuh**
(`data/ait_ads/{scenario}_wazuh.json` + `labels.csv`). Pliki Suricata EVE i AMiner
ze zbioru referencyjnego nie sa pobierane do `data/` i nie wchodza do feature buildu
(m.in. usunete zostaly kolumny `sig_id`, `sig_severity`, `sid_rev`, `src/dst_port`,
`ac_type`, `attr_value_*`, `probability_threshold`, `is_training_mode`, `group_suricata`).

## Jak odtworzyc

```bash
bash run_all.sh            # caly pipeline, logi w logs/run_all.log
```

Poszczegolne etapy (kolejnosc jest calosciowa, dane wejsciowe to `data/ait_ads/*.json` plus `labels.csv`):

```bash
.venv/bin/python -u research/build_features.py data/features      # JSON -> Parquet (2 293 628 wierszy, 78 kolumny)
.venv/bin/python -u research/report_stats.py                       # statystyki do tabel raportu
.venv/bin/python -u research/compute_stats.py                      # statystyki zrodlowe (research/ads_stats.json)
.venv/bin/python -u research/train.py --protocols los,chrono \
    --configs lgb_structured,lgb_text --deploy                     # trening i pakiet modelu
.venv/bin/python -u research/analyze_predictions.py                # analizy poreplikowe (OOF)
.venv/bin/python -u research/figures.py                            # ryciny PNG
.venv/bin/python -u research/finalize_bundle.py                    # metadata, podsumowania, checksumy
.venv/bin/python -u report/build_report.py                         # report/sprawozdanie.md
```

Wymagane pakiety w srodowisku `.venv`: pandas, pyarrow, scikit-learn, lightgbm, matplotlib, joblib, numpy.

## Artefakty

| Sciezka | Zawartosc |
| --- | --- |
| `report/sprawozdanie.md` | sprawozdanie laboratoryjne (Markdown, 8 sekcji, 5 rycin PNG) |
| `report/figures/*.png` | ryciny: F1 per scenariusz, skan progu, krzywe PR, bledy detektorow, recall faz |
| `model/model.zip` | **pakiet modelu do oddania**: `deployed_model.joblib`, `feature_plan.json`, `metadata.json`, `README.md`, `predict.py`, `requirements.txt` |
| `model/model_lgb_structured.zip`, `model/model_lgb_text.zip` | warianty konfiguracyjne tego samego pakietu |
| `model/metadata.json`, `model/thresholds.json` | plaska metadata i progi (do generatora raportu) |
| `artifacts/` | metryki per protokol i konfiguracja, progi, predykcje OOF, plan cech, analizy, checksumy |
| `research/RESEARCH.md` | research zrodlowy (publikacje, cechy, metryki, protokoly, literatura) |
| `logs/` | logi wykonania (`run_all.log`, `train.log`, `build_features.log`) |
| `data/ait_ads/`, `data/features/` | surowe JSON + cechy Parquet (duze, nie w repo) |

## Skrot wynikow (LAS = Leave-One-Scenario-Out, 8 foldow, model `lgb_structured`)

Pooled out-of-fold, przy progu wdrozeniowym 0.730463:

| Metryka | Model | Prior detektora | Zawsze-atak |
| --- | --- | --- | --- |
| F1 ataku | 0.9961 | 0.9960 | 0.8547 |
| PR-AUC | 0.9990 | 0.9983 | 0.6644 |
| Redukcja alertow (ARR) | 99.41 % | 99.41 % | 74.62 % |
| Odfiltrowane tlo | 99.95 % | 99.94 % | 0.00 % |
| FP na 1 mln alertow | 116.0 | 155.6 | 253 800 |
| Recall ataku | 0.9923 | 0.9923 | 1.0000 |

Per-scenariuszowo (F1 modelu / F1 prioru w tymze foldzie):
fox 0.9985 / 0.9985, harrison 0.9949 / 0.9948, russellmitchell 0.9608 / 0.9616,
santos 0.9171 / 0.9166, shaw 0.9329 / 0.9428, wardbeck 0.9377 / 0.9306,
wheeler 0.9989 / 0.9989, wilson 0.9959 / 0.9958.
Model rowna sie lub bije baseline w 6 na 8 foldow; przegrywa w `russellmitchell` i `shaw`.

Protokol chronologiczny (walk-forward, expanding window):

| Konfiguracja | F1 | ARR | Odfiltrowane tlo | FP / 1 mln |
| --- | --- | --- | --- | --- |
| `lgb_structured` | 0.9879 | 98.02 % | 93.49 % | 12 020 |
| `lgb_text` | 0.9901 | 98.39 % | 95.44 % | 8 414 |

Pakiet wdrozeniowy (`lgb_structured`): `model/model_lgb_structured.zip`
(141 912 B, SHA-256 `3a6aeff47e121ed9...`), `deployed_model.joblib` (19 drzew,
wczesne zatrzymanie na trzymanych 10 %), prog decyzyjny 0.730463,
budzety FP: 0.345534 (5 FP/1 mln), 0.309182 (20), 0.301409 (50).

## Uwagi metodyczne

- Etykiety sa slabe: `attack` z przedzialow czasu z `labels.csv`, fazy ataku jako etkiety wewnetrzne.
- Progi decyzyjne sa kalibrowane na zbiorze trzymanym, nie na tescie.
- Cechy to znane pola strumienia Wazuh (m.in. `rule_id`, `rule_desc`, `rule_groups`,
  `rule_level`, `decoder`, `location`, liczniki okien `-60/-600 s`, flagi grup reguly
  i MITRE, metryki HTTP, metryki logu i sciezek).
- Cecha `is_training_mode` (AMiner) jest artefiktem zbioru — w tej edycji nie wchodzi do planu cech.
- Cecha `event_source_id` jest niedostepna, wiec nie ma agregacji po ID zdarzenia.
- Uwaga o rzetelnosci: model jest mocny w scenariuszach z duzym naplywem atakow
  (fox, wheeler, wilson, harrison); najtrudniejsze sa scenariusze z przewaga tla
  (santos, shaw, wardbeck, russellmitchell) — tam FP i F1 sa najblizej progu decyzyjnego.
- Model ocenia kazdy alert indywidualnie, nie sekcje logow — w zrodle (Wazuh API)
  ten sam model ocenia alerty strumieniowo, a okna czasowe licza sie biezaco.
