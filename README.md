# Klasyfikacja alertow AIT-ADS (Wazuh + Suricata + AMiner)

Repozytorum laboratoryjne: model ML do filtrowania alertow ataku versus tlo na publicznym
zbiorze AIT Alert Data Set (AIT-ADS, DOI 10.5281/zenodo.8263181). Sprawozdanie jest w Markdownie.

## Jak odtworzyc

```bash
bash run_all.sh            # caly pipeline, logi w logs/run_all.log
```

Poszczegolne etapy (kolejnosc jest calosciowa, dane wejsciowe to `data/ait_ads/*.jsonl` plus `labels.csv`):

```bash
.venv/bin/python -u research/build_features.py data/features      # JSONL -> Parquet (2 655 821 wierszy)
.venv/bin/python -u research/report_stats.py                       # statystyki do tabel raportu
.venv/bin/python -u research/train.py --protocols los,chrono \
    --configs lgb_structured,lgb_text --deploy                     # trening i pakiet modelu
.venv/bin/python -u research/analyze_predictions.py                # analizy poreplikowe (OOF)
.venv/bin/python -u research/figures.py                            # ryciny PNG
.venv/bin/python -u research/finalize_bundle.py                    # metadata, podsumowania, checksumy
.venv/bin/python -u report/build_report.py                         # report/sprawozdanie.md
```

Wymagane poryzeki w srodowisku `.venv`: pandas, pyarrow, scikit-learn, lightgbm, matplotlib, joblib, numpy.

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
| `logs/` | logi wykonania (`run_all.log`, `train.log`, `feat.log`) |
| `data/ait_ads/`, `data/features/` | surowe JSONL + cechy Parquet (duze, nie w repo) |

## Skrot wynikow (LAS = Leave-One-Scenario-Out, 8 foldow, model `lgb_structured`)

| Metryka | Model | Priora detektora | Zawsze-atak |
| --- | --- | --- | --- |
| F1 atak | 0.9943 | 0.9938 | 0.7984 |
| PR-AUC | 0.9978 | 0.9971 | 0.7995 |
| Redukcja alertow (ARR) | 99.24 % | 99.03 % | 0.00 % |
| Fałszywe alarmy na 1 mln alertow | 0.0 | 621.7 | 335 580 |

Uwaga o rzetelnosci: model bije baseline prioru w 7 na 8 foldow, przegrywa w `wardbeck`.
Protokol chronologiczny (walk-forward) jest znacznie trudniejszy: F1 0.9904 dla 50 % i 70 % okna,
spadek do 0.85 dla 90 % okna (pojawianie sie nowych scenariuszy i faz w przeplywie).

## Uwagi metodyczne

- Etkiety sa slabe: `attack` z przedzialow czasu z `labels.csv`, fazy ataku jako etkiety wewnetrzne.
- Progi decyzyjne sa kalibrowane na zbiorze trzymanym, nie na teście.
- Cechy to znane pola AIT-ADS (m.in. `log_column`, `AnalysisComponentName`, `rule_desc`, `sig_id`, pola HTTP);
  `event_source_id` jest niedostepny, wiec nie ma agregacji po ID zdarzenia.
- Cecha `is_training_mode` (AMiner, 4739 alertow) jest artefiktem zbioru — wdrozenie uczy sie jej na calym zbiorze.
- Cechy binarne `protocol_110/118` (SMTP, PUP) sa usuwane, bo wystepuja tylko w jednym scenariuszu.

## Struktura kodu

```
research/ml_common.py        sluzby: ladowanie cech, plan cech, metryki, protokoly, LightGBM
research/build_features.py   strumieniowy JSONL -> Parquet (batchowo, 2,7 GB -> 65 MB)
research/train.py            LAS + chronologiczny + pakiet wdrożeniowy (z ZIP-a)
research/analyze_predictions.py  analizy poreplikowe: detektory, fzy, skan progu, novel-detektory
research/figures.py          ryciny PNG
research/finalize_bundle.py  metadata, podsumowania, sumy kontrolne SHA-256
report/build_report.py       generator sprawozdania (Markdown)
report/sprawozdanie.md       sprawozdanie laboratoryjne
model_assets/predict.py      przykladowa inference (otwiera zip, czyta plan cech, zapisuje predykcje)
```
