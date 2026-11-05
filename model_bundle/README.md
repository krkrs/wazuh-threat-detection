# Pakiet modelu — AIT-ADS, klasyfikacja alertów (atak / tło)

Zawartość archiwum `model.zip` (po odpakowaniu):

```text
model/deployed_model.joblib   model LightGBM + encodery tekstowe (TF-IDF + TruncatedSVD) dla wariantu tekstowego
model/feature_plan.json       listy cech liczbowych/kategorycznych, porządek kolumn, cechy odrzucone
model/metadata.json           konfiguracja, próg decyzyjny, metryki na trzymanej próbie, sumy kontrolne
model/predict.py              skrypt inferencji (pandas + lightgbm + scikit-learn + joblib)
model/requirements.txt        zależności do uruchomienia inferencji
model/README.md               ta instrukcja
```

## Szybki start

```bash
unzip model.zip -d model_bundle
python -m venv .venv-infer && .venv-infer/bin/pip install -r model_bundle/model/requirements.txt
.venv-infer/bin/python model_bundle/model/predict.py \
    --features data/features/fox.features.parquet \
    --out scores/fox.csv \
    --bundle model_bundle \
    --evaluate
```

Wejściem jest **tabela cech** (Parquet) wyprodukowana przez `research/build_features.py`
(152 kolumn, jeden wiersz = jeden alert). Skrypt odtwarza konstruowanie macierzy cech
identycznie jak w treningu (cechy liczbowe `float32`, kategoryczne `category`,
wektor tekstowy TF-IDF+SVD tylko dla wariantu tekstowego), a następnie zapisuje
`prob_attack` i `pred_attack` (1 = eskaluj do analityka, 0 = odfiltruj jako tło).

`--evaluate` wypisuje precision/recall/F1/ARR, jeśli tabela cech zawiera kolumnę `label`.
`--threshold` nadpisuje zapisany próg decyzyjny (patrz tabela operating pointów w sprawozdaniu).

## Parametry modelu (plik `model/metadata.json`)

- zadanie: klasyfikacja binarna `attack` vs `background` na poziomie alertu,
- dane: 8 scenariuszy AIT-ADS (DOI 10.5281/zenodo.8263181), 2 655 821 alertów,
- split wdrożeniowy: losowe 90/10 (seed 42); wczesne zatrzymywanie liczone na tej samej
  trzymanej próbie 10 %, dlatego metryki w `metadata.json` są **optymistyczne**,
- nieobciążone oszacowania: protokół Leave-One-Scenario-Out i chronologiczny walk-forward
  (`artifacts/metrics_*.json` w repozytorium źródłowym),
- próg decyzyjny: wybrany jako optimum F1 na scenariuszu kalibracyjnym (LAS),
- features: 82 liczbowe + 25 kategoryczne (+24 komponenty SVD dla wariantu tekstowego).

## Ograniczenia

- Inferencja wymaga **tych samych kolumn** co tabela treningowa — nowy alert musi przejść
  przez `build_features.py` (liczniki kontekstowe 60 s / 600 s są liczone po `timestamp`).
- Model nie zna detektorów spoza słownika 93 kluczów użytych w treningu: dla nowej reguły
  LightGBM traktuje kategorię jako nieznaną (brak kodu) i opiera decyzję na cechach
  kontekstowych — przy dużej liczbie nowych reguł zalecany jest retrening.
- Etykiety ataku w danych pochodzą z przedziałów czasowych `labels.csv` (słabe etykiety),
  więc recall mierzy zgodność z tą definicją, nie zdolność wykrycia każdego realnego ataku.
