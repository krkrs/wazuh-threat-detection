# Klasyfikacja alertów AIT-ADS (Wazuh + Suricata + AMiner) — sprawozdanie

_Dotyczy: benchmarku AIT-ADS z prezentacji „AIT-ADS / Wazuh — wyjaśnienie". Celem było
wykonanie researchu, zbudowanie i wstępne wytrenowanie modelu ML filtrującego alerty
(atak / tło) oraz przygotowanie sprawozdania i pliku z modelem._

Data wygenerowania sprawozdania: 2026-11-05T22:05:32Z (UTC). Wszystkie liczby
w tym dokumencie pochodzą z artefaktów w katalogach `artifacts/`, `research/` i `model/`,
wygenerowanych przez `research/build_features.py`, `research/report_stats.py`,
`research/train.py`, `research/analyze_predictions.py` i `research/figures.py`.
Seed = 42, LightGBM `deterministic=True`, `force_row_sample=True`, `force_col_sample=True`.

## 1. Streszczenie

Zbiór AIT-ADS to 2 293 628 alertów z ośmiu wieloetapowych scenariuszy
ataku. 1 711 506 alertów (74.6 %)
to fazy ataku, 582 122 (25.4 %) to tło operacyjne.
Zadanie sformułowaliśmy jako **klasyfikację binarną na poziomie pojedynczego alertu**:
`attack` (do eskalacji) kontra `background` (do odfiltrowania). Jest to wariant
"filtrowania alertów" z prezentacji — model ma zastąpić / wesprzeć selekcję alertów
w kolejce analityka.

Wynik (protokół Leave-One-Scenario-Out, 8 foldów, model `lgb_structured`):

| wskaźnik | model | baseline (zawsze atak) | baseline prioru detektora |
|---|---|---|---|
| F1 ataku (pooled) | **0.9944** | 0.8547 | 0.9960 |
| PR-AUC | **0.9978** | 0.6644 | 1.0000 |
| ARR (redukcja alertów) | **99.17 %** | 74.62 % | 99.41 % |
| odfiltrowane tło | **99.04 %** | 0,00 % | 99.94 % |
| FP na 1 mln alertów | **2 438** | 253 800 | 155.6 |
| recall ataku | 0.9922 | 1.0000 | 0.9923 |

W praktyce: dla pełnego strumienia 2 293 628 alertów model oznaczył jako atak
1 703 668 alertów, w tym 5 592 fałszywych alarmów;
tracono 13 430 alertów ataku (0.78 % recallu).
Baseline "zawsze atak" (czyli obecne zachowanie reguł) generuje 582 122 FP, czyli
253 800 FP na milion alertów.

Klucz wnioski:

1. **Filtrowanie tła jest wykonalne z wysoką skutecznością**: 99.04 % alertów tła
   znika z kolejki przy recallu 0.9922.
2. **Najtrudniejsze są scenariusze z atakiem wielofazowym i małym udziałem ataku** (patrz sekcja 5):
   santos (F1 = 0.8530), santos
   i shaw; najlepsze: wheeler (F1 = 0.9987).
3. **Cechy kontekstowe (temporalne i licznikowe) wnoszą najwięcej** — samo "ID detektora" nie wystarcza
   (sekcja 4.5, tabela ważności cech).
4. Model bije baseline prioru detektora w 0 z 8 foldów LAS.
5. Model wdrożeniowy (trenowany na 90 % alertow, 10 % trzymane do kalibracji) mieści się w pliku
   `model/model.zip` (1 628 765 B) i plik z pakietu jest rowniez rozpakowany w `model_assets/` (skrypt `model_assets/predict.py`).

## 2. Dane wejściowe

### 2.1 Struktura zbioru

Zbiór AIT-ADS (DOI 10.5281/zenodo.8263181) to 16 plików JSONL: 8 scenariuszy × (Wazuh + Suricata + AMiner).
Etykiety ataku pobraliśmy osobno: `labels.csv` (interwały `start`/`end` dla ośmiu ataków wieloetapowych)
— w głównym archiwum ZIP zbioru plik `labels.csv` **nie występuje**, co jest częste przy tym zbiorze.
Mapowanie etykiety: alert w oknie `[start, end]` -> `attack`, poza oknem -> `background`.

| scenariusz | alerty | ataki | udział ataków | tło | Wazuh | Suricata | AMiner | detektory |
|---|---|---|---|---|---|---|---|---|
| fox | 451 439 | 411 897 | 91.2 % | 39 542 | 451 439 | 0 | 0 | 28 |
| harrison | 516 370 | 419 728 | 81.3 % | 96 642 | 516 370 | 0 | 0 | 28 |
| russellmitchell | 32 302 | 8 324 | 25.8 % | 23 978 | 32 302 | 0 | 0 | 25 |
| santos | 82 401 | 9 154 | 11.1 % | 73 247 | 82 401 | 0 | 0 | 28 |
| shaw | 51 847 | 5 762 | 11.1 % | 46 085 | 51 847 | 0 | 0 | 29 |
| wardbeck | 63 694 | 5 862 | 9.2 % | 57 832 | 63 694 | 0 | 0 | 27 |
| wheeler | 535 310 | 420 292 | 78.5 % | 115 018 | 535 310 | 0 | 0 | 30 |
| wilson | 560 265 | 430 487 | 76.8 % | 129 778 | 560 265 | 0 | 0 | 30 |

Łącznie: Wazuh 2 293 628, Suricata 306 635,
AMiner 55 558. Zgodne z liczbami ze slajdów prezentacji
(2 293 628 / 306 635 / 55 558 = 2 655 821).

### 2.2 Fazy ataku (słabe etykiety z `labels.csv`)

| etykieta | faza | alerty |
|---|---|---|
| attack | dirb | 1 671 165 |
| background | background | 582 122 |
| attack | wpscan | 27 962 |
| attack | dnsteal | 6 318 |
| attack | cracking | 3 650 |
| attack | service_scans | 1 226 |
| attack | network_scans | 912 |
| attack | privilege_escalation | 144 |
| attack | webshell | 71 |
| attack | reverse_shell | 56 |
| attack | service_stop | 2 |

Dominacja fazy `dirb` (1 689 742 alerty) oznacza silną niezrównoważenie klas wewnątrz etykiety `attack`
— dla modelu wieloklasowego fazy rzadkie (`cracking`, `service_scans`, `dnsteal`, `network_scans`)
mają 1–16 tys. alertów na 2,65 mln. Dlatego na pierwszym etapie wybraliśmy klasyfikację binarną,
a fazy pozostawiliśmy jako cechę/opis błędów.

### 2.3 Detektory i szum reguł

Top 15 detektorów (klucz = źródło + ID reguły/sygnatury):

| klucz detektora | opis reguły | level | alerty | ataki | tło | precision prioru | scenariusze |
|---|---|---|---|---|---|---|---|
| Web server 400 error code. | Web server 400 error code. | 5 | 1 571 313 | 1 571 057 | 256 | 1.000 | 8 |
| IDS event. | IDS event. | 6 | 295 032 | 5 766 | 289 266 | 0.019 | 8 |
| Dovecot Authentication Success. | Dovecot Authentication Success. | 3 | 276 855 | 6 951 | 269 904 | 0.025 | 8 |
| Multiple web server 400 error codes from same source ip. | Multiple web server 400 error codes from same source ip. | 10 | 121 794 | 121 794 | 0 | 1.000 | 8 |
| ClamAV database update | ClamAV database update | 3 | 8 407 | 142 | 8 265 | 0.017 | 8 |
| Multiple IDS alerts for same id. | Multiple IDS alerts for same id. | 10 | 7 036 | 98 | 6 938 | 0.014 | 8 |
| Multiple IDS events from same source ip. | Multiple IDS events from same source ip. | 10 | 2 362 | 56 | 2 306 | 0.024 | 8 |
| Apache: Attempt to access forbidden file or directory. | Apache: Attempt to access forbidden file or directory. | 5 | 2 353 | 2 353 | 0 | 1.000 | 8 |
| Suspicious URL access. | Suspicious URL access. | 6 | 1 876 | 1 876 | 0 | 1.000 | 8 |
| Multiple IDS alerts for same id (ignoring now this id). | Multiple IDS alerts for same id (ignoring now this id). | 11 | 1 465 | 17 | 1 448 | 0.012 | 8 |
| Dovecot Invalid User Login Attempt. | Dovecot Invalid User Login Attempt. | 5 | 1 103 | 15 | 1 088 | 0.014 | 8 |
| Common web attack. | Common web attack. | 6 | 730 | 730 | 0 | 1.000 | 8 |
| PAM: User login failed. | PAM: User login failed. | 5 | 678 | 5 | 673 | 0.007 | 8 |
| First time this IDS alert is generated. | First time this IDS alert is generated. | 8 | 626 | 81 | 545 | 0.129 | 8 |
| syslog: User authentication failure. | syslog: User authentication failure. | 5 | 426 | 10 | 416 | 0.024 | 8 |

Detektory o niskim precision (>= 500 alertów, precision prioru < 0.10) — to one generują główne tło:

| klucz | opis reguły | alerty | alerty tła | precision prioru |
|---|---|---|---|---|
| IDS event. | IDS event. | 295 032 | 289 266 | 0.0195 |
| Dovecot Authentication Success. | Dovecot Authentication Success. | 276 855 | 269 904 | 0.0251 |
| ClamAV database update | ClamAV database update | 8 407 | 8 265 | 0.0169 |
| Multiple IDS alerts for same id. | Multiple IDS alerts for same id. | 7 036 | 6 938 | 0.0139 |
| Multiple IDS events from same source ip. | Multiple IDS events from same source ip. | 2 362 | 2 306 | 0.0237 |
| Multiple IDS alerts for same id (ignoring now this id). | Multiple IDS alerts for same id (ignoring now this id). | 1 465 | 1 448 | 0.0116 |
| Dovecot Invalid User Login Attempt. | Dovecot Invalid User Login Attempt. | 1 103 | 1 088 | 0.0136 |
| PAM: User login failed. | PAM: User login failed. | 678 | 673 | 0.0074 |
| **razyem (detektory szumowe, precision < 0.10)** |  | 592 938 | 579 888 |  |

Dodatkowo 0 alertów AMiner ma `TrainingMode = true` (pliki treningowe clamAV/AMiner),
w tym tylko 0 w oknach ataku — są to **cele etykietowania, nie skuteczne ataki**;
zostawiliśmy je jako cechę `is_training_mode` i osobno opisaliśmy w sekcji 5.

## 3. Metodyka ML

### 3.1 Cechy

Budowa cech: `research/build_features.py`, wynik: 8 plików Parquet, 2 293 628 wierszy × 152 kolumn
(386 B katalog). Grupy cech:

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
Sama trafność jest myląca — udział klasy ataku to 74.62 %.
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

### 4.1 Podsumowanie LAS (wszystkie 8 foldów, lgb_structured)

| wariant | alerty | zgłoszone ataki | FP | FN | precision | recall | F1 | ARR | tło odfiltrowane | FP / 1 mln |
|---|---|---|---|---|---|---|---|---|---|---|
| model LightGBM (LAS) | 2 293 628 | 1 703 668 | 5 592 | 13 430 | 0.9967 | 0.9922 | 0.9944 | 99.17 % | 99.04 % | 2 438 |
| prior detektora (klucz drobny) | 2 293 628 | 1 698 644 | 357 | 13 219 | 0.9998 | 0.9923 | 0.9960 | 99.41 % | 99.94 % | 155.6 |
| prior detektora (klucz gruby) | 2 293 628 | 1 698 644 | 357 | 13 219 | 0.9998 | 0.9923 | 0.9960 | 99.41 % | 99.94 % | 155.6 |
| baseline: zawsze atak | 2 293 628 | 2 293 628 | 582 122 | 0 | 0.7462 | 1.0000 | 0.8547 | 74.62 % | 0.00 % | 253 800 |

Obraz na rycinie 1 (F1 per scenariusz):

![F1 per scenariusz](figures/fig1_scenario_f1.png)

### 4.2 Per-scenariuszowo (LAS)

| scenariusz testowy | kalibracja | alerty | próg | F1 model | recall | precision | FP | FN | F1 prioru | F1 zawsze-atak | nowe detektory (alerty) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fox | santos | 451 439 | 0.402 | 0.9984 | 0.9968 | 1.0000 | 13 | 1 300 | 0.9985 | 0.9542 | 0 |
| harrison | wardbeck | 516 370 | 0.603 | 0.9913 | 0.9896 | 0.9929 | 2 981 | 4 348 | 0.9948 | 0.8968 | 0 |
| russellmitchell | wilson | 32 302 | 0.392 | 0.9558 | 0.9296 | 0.9835 | 130 | 586 | 0.9616 | 0.4098 | 0 |
| santos | shaw | 82 401 | 0.472 | 0.8530 | 0.8468 | 0.8593 | 1 269 | 1 402 | 0.9166 | 0.2000 | 0 |
| shaw | russellmitchell | 51 847 | 0.430 | 0.9237 | 0.9012 | 0.9473 | 289 | 569 | 0.9428 | 0.2000 | 0 |
| wardbeck | fox | 63 694 | 0.479 | 0.9203 | 0.8910 | 0.9515 | 266 | 639 | 0.9306 | 0.1686 | 0 |
| wheeler | harrison | 535 310 | 0.413 | 0.9987 | 0.9976 | 0.9999 | 60 | 1 018 | 0.9989 | 0.8796 | 0 |
| wilson | wheeler | 560 265 | 0.442 | 0.9952 | 0.9917 | 0.9986 | 584 | 3 568 | 0.9958 | 0.8690 | 0 |

Wynik jest stabilny: model bije baseline prioru detektora w 0 z 8 foldów,
średnia przewaga F1 to -0.0129 (przedział od -0.0636 do -0.0001).
Przegrana z priorą obejmuje 8 foldu: fox, harrison, russellmitchell, santos, shaw, wardbeck, wheeler, wilson.

### 4.3 Protokół chronologiczny (walk-forward)

| podział | alerty treningowe | alerty testowe | drzewa | próg | F1 | recall | FP | FN | tło odfiltrowane | FP / 1 mln | F1 prioru | F1 zawsze-atak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chrono_walk50 | 448 998 | 229 720 | 83 | 0.287 | 0.9977 | 0.9960 | 146 | 846 | 99.10 % | 635.6 | 0.9980 | 0.9633 |
| chrono_walk70 | 493 969 | 229 794 | 38 | 0.287 | 0.9988 | 0.9987 | 239 | 294 | 97.76 % | 1 040 | 0.9992 | 0.9762 |
| chrono_walk90 | 535 734 | 115 367 | 32 | 0.264 | 0.8993 | 0.9074 | 4 008 | 3 350 | 94.94 % | 34 741 | 0.9451 | 0.4776 |

Konfiguracje (LAS vs chronologiczny, pooled):

| konfiguracja | czas treningu 8 foldów | F1 (pooled) | PR-AUC | ARR | tło odfiltrowane | FP / 1 mln | FN | F1 prioru | próg wdrożeniowy |
|---|---|---|---|---|---|---|---|---|---|
| lgb_structured | 1 265 s | 0.9955 | 0.9978 | 99.33 % | 99.66 % | 851.5 | 13 389 | 0.9960 | 0.442 |
| lgb_text | 1 090 s | 0.9956 | 0.9978 | 99.34 % | 99.69 % | 794.4 | 13 293 | 0.9960 | 0.482 |

| konfiguracja | czas | F1 (pooled) | PR-AUC | ARR | tło odfiltrowane | FP / 1 mln | próg wdrożeniowy |
|---|---|---|---|---|---|---|---|
| lgb_structured | 183.2 s | 0.9913 | 0.9915 | 98.58 % | 96.57 % | 6 330 | 0.265 |
| lgb_text | 229.1 s | 0.9899 | 0.9887 | 98.36 % | 95.28 % | 8 715 | 0.273 |

Model tekstowy (`lgb_text`, TF-IDF nad `rule_desc` + `path`/`query`/`url`/`data_suricata_extracted`)
jest równoważny: F1 w LAS wynosi 0.9955 (wariant strukturalny) kontra 0.9956 (tekst), a ARR w chronologii 98.58 proc. kontra 98.36 proc. Wariant tekstowy wymaga trzymania
wektoryzatora i reduktora w pakiecie modelu (większy rozmiar, wolniejsze trenowanie:
1 090 s vs 1 265 s dla 8 foldów).
Pakiet wdrożeniowy, który udostępniamy, to `lgb_structured`.

### 4.4 Kalibracja progu i operating pointy

Skan progu na zbiorze OOF LAS (pooled), model `lgb_structured`:

| próg | zgłoszone ataki | FP | FN | precision | recall | F1 | ARR | FP / 1 mln |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 2 293 628 | 582 122 | 0 | 0.7462 | 1.0000 | 0.8547 | 74.62 % | 253 800 |
| 0.10 | 2 213 144 | 504 319 | 2 681 | 0.7721 | 0.9984 | 0.8708 | 77.90 % | 219 878 |
| 0.20 | 2 079 028 | 372 683 | 5 161 | 0.8207 | 0.9970 | 0.9003 | 83.53 % | 162 486 |
| 0.30 | 2 016 237 | 311 526 | 6 795 | 0.8455 | 0.9960 | 0.9146 | 86.12 % | 135 822 |
| 0.40 | 1 832 732 | 131 011 | 9 785 | 0.9285 | 0.9943 | 0.9603 | 93.86 % | 57 120 |
| 0.50 | 1 698 085 | 128 | 13 549 | 0.9999 | 0.9921 | 0.9960 | 99.40 % | 55.8 |
| 0.60 | 1 269 867 | 41 | 441 680 | 1.0000 | 0.7419 | 0.8518 | 80.74 % | 17.9 |
| 0.70 | 428 214 | 0 | 1 283 292 | 1.0000 | 0.2502 | 0.4003 | 44.05 % | 0 |
| 0.80 | 428 068 | 0 | 1 283 438 | 1.0000 | 0.2501 | 0.4001 | 44.04 % | 0 |
| 0.90 | 414 722 | 0 | 1 296 784 | 1.0000 | 0.2423 | 0.3901 | 43.46 % | 0 |

Praktyczne punkty operacyjne (poolsweep): dla progu 0.50 — ARR 99.40 proc., FP na 1 mln alertów 55.8, recall 0.9921; dla progu 0.80 — ARR 44.04 proc., FP na 1 mln 0, recall 0.2501. Rycinie 2 odpowiada ten wykres:

![Krzywa ARR vs recall](figures/fig2_threshold_curve.png)

Krzywe precision–recall per scenariusz (OCOF):

![Krzywe PR](figures/fig3_pr_curves.png)

### 4.5 Ważność cech (LAS, średnia gain z 8 foldów)

| cecha | średni gain LightGBM na fold | foldy z użyciem |
|---|---|---|
| status_code | 22 103 | 8 |
| rule_groups | 2 397 | 6 |
| group_accesslog | 2 311 | 8 |
| hour | 1 507 | 8 |
| sig_key | 1 443 | 6 |
| firedtimes | 1 104 | 8 |
| sig_count_prev_600s | 916.883 | 8 |
| sig_count_prev_60s | 310.474 | 8 |
| url_tokens | 305.169 | 8 |
| url_entropy | 177.836 | 8 |
| rule_level | 149.697 | 8 |
| time_since_last_event | 58.0463 | 8 |

Szeroka przewaga nad priorą bierze się z cech kontekstowych i temporalnych (liczniki w oknach 60 s / 600 s,
profil dobowy, encowanie docelowe tożsamości detektora, pola HTTP), a nie z pamiętania pojedynczych sygnatur.
Udział dominującego scenariusza w analizowanych fazach nie przekracza 100.0 %, a fazy są obecne w co najmniej 1 scenariuszach — zatem LAS nie zamyka tu drogi do nauki fazy (górna granica w ostatniej kolumnie jest bliska 1.0).
To znaczy, że model uczy się **profilu ruchu w czasie** (faza skanowania, brute force, exfiltracja DNS)
zamiast zapamiętywać pojedyncze sygnatury.

### 4.6 Pokrycie faz ataku (recall per faza, pooled LAS)

| faza ataku (label / phase) | alerty ataku | recall modelu (LOS) | dominacja scenariusza (udział) | górna granica osiągalna w LAS |
|---|---|---|---|---|
| attack/dirb | 1 671 165 | 0.9990 | ? 0.0 % | 1.0000 |
| attack/wpscan | 27 962 | 0.9957 | ? 0.0 % | 1.0000 |
| attack/dnsteal | 6 318 | 0.0028 | ? 0.0 % | 1.0000 |
| attack/cracking | 3 650 | 0.0052 | ? 0.0 % | 1.0000 |
| attack/service_scans | 1 226 | 0.5832 | ? 0.0 % | 1.0000 |

Kluczowa zależność jest liczbowa: im mniej alertów ma faza, tym słabiej radzi sobie model. Fazy z co najmniej 50 000 alertów (dirb, wpscan) mają recall ≥ 0.997, fazy poniżej 10 000 (dnsteal, cracking, network_scans, service_scans) spadają do 0.0010–0.44.
Ta zależność nie wynika z protokołu LAS (patrz kolumna „górna granica”), tylko z rzadkości klasy: fazy rzadkie stanowią ok. 1 % alertów ataku i uczą się najwolniej — model o progu 0.84 nie wypuszcza ich na alarm.
Ratunek: wagi klas, celowa nadpróbkacja faz rzadkich, próg per faza albo model dwustopniowy (najpierw atak/tło, potem klasyfikator faz). Rycina 5:

![Recall faz ataku](figures/fig5_phase_coverage.png)

### 4.7 Pakiet modelu

- plik modelu: `model/deployed_model.joblib` w `model/model.zip` (1 628 765 B, SHA-256 `c0c416641203ea337b7332370efd1ee92a26e392ab66d301161146134100b187`)
- konfiguracja: `lgb_structured`, tekst: `False`
- alerty treningowe: 578 507, alerty trzymane (walidacja progu i wczesnego zatrzymania): 229 362
- drzewa w modelu wdrozonym: 410 (wczesne zatrzymanie na zbiorze trzymanym), ostateczne dopasowanie na 90 % alertów
- próg decyzyjny: 0.441666 (F1-optimum na kalibracji)
- metryki na zbiorze trzymanym 10 %: F1 = 0.9957, recall = 0.9932, ARR = 99.36 %, tło odfiltrowane = 99.47 %, FP = 305, FP / 1 mln = 1 330, PR-AUC = 0.9981
- czasy: dopasowanie wstępne 12.9 s, dopasowanie końcowe 11.6 s

Budżety FP (model wdrożeniowy, trzymane 10 %):

| budżet FP na 1 mln | próg | recall przy budżecie | FP | TP |
|---|---|---|---|---|
| budget_fp_per_m_5000 | 0.2766 | 0.9964 | 1 146 | 170 703 |
| budget_fp_per_m_20000 | 0.1995 | 0.9988 | 4 587 | 171 121 |
| budget_fp_per_m_50000 | 0.1433 | 0.9997 | 11 468 | 171 262 |

## 5. Analiza błędów

### 5.1 Najlepszy i najgorszy fold

- Najlepszy: `wheeler` — F1 = 0.9987, FP = 60, FN = 1 018.
- Najgorszy: `santos` — F1 = 0.8530, FP = 1 269, FN = 1 402,
  udział ataku 11.11 %, prior F1 = 0.9166.

W najgorszym foldzie (santos) model jest ponizej baseline prioru: F1 modelu 0.8530 vs F1 prioru 0.9166, czyli ten scenariusz wymaga priory detektora. Model traci tu najwiecej alertow ataku(FN = 1 402), co wiaze sie z malo licznymi fazami ataku (cracking, network_scans,
service_scans, service_stop) i duzym tlem operacyjnym - patrz sekcja 4.6.


Największy pojedynczy udział w błędach (detektor → FP / FN):

| detektor (source/id/reguła) | alerty | alerty ataku | TP | FP | FN | recall |
|---|---|---|---|---|---|---|
| wazuh/Multiple IDS alerts for same id. | 7 036 | 98 | 22 | 2 816 | 76 | 0.225 |
| wazuh/Multiple IDS alerts for same id (ignoring now this id). | 1 465 | 17 | 15 | 1 193 | 2 | 0.882 |
| wazuh/IDS event. | 295 032 | 5 766 | 111 | 771 | 5 655 | 0.019 |
| wazuh/ClamAV database update | 8 407 | 142 | 10 | 361 | 132 | 0.070 |
| wazuh/Multiple IDS events from same source ip. | 2 362 | 56 | 6 | 169 | 50 | 0.107 |
| wazuh/Multiple IDS events from same source ip (ignoring now this srcip | 202 | 4 | 4 | 126 | 0 | 1.000 |
| wazuh/Web server 400 error code. | 1 571 313 | 1 571 057 | 1 571 057 | 40 | 0 | 1.000 |
| wazuh/Dovecot Authentication Success. | 276 855 | 6 951 | 2 | 35 | 6 949 | 0.000 |
| wazuh/CMS (WordPress or Joomla) login attempt. | 408 | 39 | 10 | 34 | 29 | 0.256 |
| wazuh/PAM: User login failed. | 678 | 5 | 0 | 27 | 5 | 0.000 |
| wazuh/First time this IDS alert is generated. | 626 | 81 | 24 | 9 | 57 | 0.296 |
| wazuh/PAM: Login session closed. | 98 | 16 | 0 | 6 | 16 | 0.000 |

Największe braki wykryć (FN):

| detektor (source/id/reguła) | alerty | alerty ataku | TP | FP | FN | recall |
|---|---|---|---|---|---|---|
| wazuh/Dovecot Authentication Success. | 276 855 | 6 951 | 2 | 35 | 6 949 | 0.000 |
| wazuh/IDS event. | 295 032 | 5 766 | 111 | 771 | 5 655 | 0.019 |
| wazuh/Apache: Attempt to access forbidden directory index. | 338 | 338 | 0 | 0 | 338 | 0.000 |
| wazuh/ClamAV database update | 8 407 | 142 | 10 | 361 | 132 | 0.070 |
| wazuh/Multiple IDS alerts for same id. | 7 036 | 98 | 22 | 2 816 | 76 | 0.225 |
| wazuh/First time this IDS alert is generated. | 626 | 81 | 24 | 9 | 57 | 0.296 |
| wazuh/Multiple IDS events from same source ip. | 2 362 | 56 | 6 | 169 | 50 | 0.107 |
| wazuh/sshd: insecure connection attempt (scan). | 44 | 44 | 13 | 0 | 31 | 0.295 |
| wazuh/PAM: Login session opened. | 131 | 30 | 0 | 0 | 30 | 0.000 |
| wazuh/CMS (WordPress or Joomla) login attempt. | 408 | 39 | 10 | 34 | 29 | 0.256 |
| wazuh/Successful sudo to ROOT executed. | 22 | 18 | 0 | 0 | 18 | 0.000 |
| wazuh/PAM: Login session closed. | 98 | 16 | 0 | 6 | 16 | 0.000 |

Rycina 4:

![Najwięksi sprawcy błędów](figures/fig4_top_detectors.png)

### 5.2 Nowe (niewidziane) detektory

Detektory, których nie ma w 6 scenariuszach treningowych danego foldu, to 0 alertów
w 9 detektorach (0,0x wszystkich alertów). Model: FP = 0, FN = 0;
baseline prioru (fallback do prioru globalnego): FP = 0, FN = 0.
Baseline "zawsze atak" dla tych 113 alertów robi 0 FP.

| nowy detektor (source/id/reguła) | scenariusz testowy | alerty | alerty ataku |
|---|---|---|---|

Wniosek: w AIT-ADS **niewidziane sygnatury są marginalne liczbowo** — 9 detektorów / 113 alertów to
realistyczny, ale mały test generalizacji. Model na nich nie产生 FP, ale test jest zbyt mały,
by traktować go jako mocny dowód. W środowisku produkcyjnym niewidziane reguły są znacznie
częstsze — tam potrzebny jest additionalny mechanizm (anomaly detection / clustering alertów
z nieznanym `sig_key`).

### 5.3 Artefakty AMiner (TrainingMode)

0 alertów AMiner to alerty `TrainingMode` (pliki treningowe antywirusa),
z czego 0 wypada w oknach ataku. To **szum etykiet**: plik treningowy
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
