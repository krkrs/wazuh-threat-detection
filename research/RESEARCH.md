# Research: AIT-ADS + Wazuh — rozwiązania zadania laboratoryjnego (model ML, wstępny trening, sprawozdanie PDF, model ZIP)

Stan: 2025-11-01, katalog roboczy `/home/krkrs/wazuh-3`, Python 3.13.5, 16 rdzeni CPU, brak GPU.
Pliki wykonawcze tego researchu: `research/compute_stats.py` (parser statystyczny), `research/ads_stats.json` (wyniki pomiarów), `research/ads_stats.err`.

---

## 0. Wypunktowanie — rekomendacja

1. **Zadanie ML**: binarna klasyfikacja `attack` vs `background` na całym zbiorze AIT-ADS (2 655 821 alertów: Wazuh + Suricata + AMiner), z etykietą `time_label` wyliczoną z interwałów faz ataku (`labels.csv`). To dokładnie ten protół, którego używają autorzy benchmarku i prace, które są “kolejnym laboratorium”.
2. **Model**: baseline = “prior detektora” (tylko `rule.id` / `AnalysisComponentName`); model główny = gradient boosting na tablicy cech kategorialnych i czasowych (`LightGBM` lub `HistGradientBoosting`; opcjonalnie XGBoost-Focal, wiernie odtwarza basen z literatury).
3. **Podział danych**: leave-one-scenario-out (8 foldów) oraz osobno podział chronologiczny (expanding window). Randomowy split po alertach nie jest miarodajny — przecieka te same sygnacje detektorów i te same hosty.
4. **Metryki**: precision / recall / F1, PR-AUC (average precision), FPC, FP na 1 mln alertów tła, oraz Alert Reduction Rate (ARR) — oficjowa metryka redukcji alertów w AIT-ADS. Raport per scenariusz i per detektor.
5. **Artefamy laboratoryjne**: `model/model.zip` (model + preprocessing + `metadata.json` + `predict.py`) i `report/sprawozdanie.pdf` (układ sekcji w §10).

Wstępne pomiary (§3) pokazują, że dataset jest silnie “feature-gęsty”: 10 sygnacji z precision ≥ 0,5 obejmuje 1 746 992 alertów (65,8% zbioru), a 25 sygnacji z precision < 0,5 generuje 905 759 alertów (34,1%), z czego 886 619 to tło. To oznacza, że sensowny basen tabularny jest osiągalny, a najważniejszą pracą jest pilnowanie przeciążenia metryk i poprawnego podziału, a nie “trudny model”.

## 1. Co realnie dostaje laboratorium (stanowisko zweryfikowane, nie z prezentacji)

Utworzone: `data/ait-alerts-ait-ads-23-1.zip` (pobrany) i `data/ait_ads/` (rozpakowany).

| Informacja | Zweryfikowany stan |
|---|---|
| Rozmiar `ait-alerts-ait-ads-23-1.zip` | 96 MB (nie 115 MB z notatek laboratoryjnych), SHA-256 `a188201780293551822417262674873856304383217602504208614033543011` |
| Zawartość | 16 plików: `{scenario}_wazuh.json` (8) i `{scenario}_aminer.json` (8). Brak osobnych `*_suricata_eve.json`, `*_zeek_*`, `suricata_stats.log` — to są elementy `full-ait-ads.zip` (1,1 GB) oraz konfiguracji `wazuh_suricata_config_export.zip` (185 MB). Do pracy na alertach wystarczy zip 96 MB. |
| Po rozpakowaniu | 2,7 GB, 16 plików JSONL. Największe: `harrison_wazuh.json` 632 MB, `wilson_wazuh.json` 616 MB, `fox_wazuh.json` 491 MB. |
| Liczby alertów | zgodne z prezentacją: Wazuh 2 293 628, Suricata 306 635 (w tym zbiorze Suricata jest widoczne wyłącznie przez Wazuh, po `rule.description` zaczynającym się od `Suricata:`), AMiner 55 558, razem 2 655 821 — potwierdzone moim parserem. |
| `labels.csv` | nie ma go w zipie — trzeba go pobrać osobno z Zenodo (3 703 B, 79 wierszy: `scenario,attack,start,end`). To pierwszy realny problem zadania; plik jest już w `data/ait_ads/labels.csv`. |
| Format Wazuh | JSON: `@timestamp` (ISO), `full_log`, `decoder.name`, `predecoder.{program_name,hostname,timestamp}`, `agent.{id,name,ip}`, `rule.{id,level,description,groups}`, `rule.mitre.tactic|technique`, `location` (np. `/var/log/apache2/access.log`), `data.{srcip,dstport,srcuser,url}`, a w warstwy z tabel ML także `field.detection` (wektor cech) i `field.decision` (`true_positive`). |
| Format Surikiaty przez Wazuh | `rule.groups: ["suricata","ids"]`, `suricata.{category,severity,signature_id,app_protocol,alert.signature,action}`, część alertów ma pola MITRE. |
| Format AMiner | JSONL: `AnalysisComponent.AnalysisComponentName`, `AnalysisComponentDescription`, `AffectedLogAtomPaths`, `AffectedLogAtomID`, `LogData.DetectionTimestamp` (epoch sekundy, bez metki czasu w nazwie — częsty bąd przy parsowaniu), `LogData.Events[].TrainingMode`, `LogData.Events[].Data.<id>.frequency`, `AnalysisContext`, `ExplanationStrings`, `NotificationReport`, `LogType`, `AIT:DocId`, `AIT:LogAtomLCREntityID`. |
| Warianty zbioru | `filtered-ait-alerts-23-01-11.zip` i `filtered-ait-ads-ml-tables.zip` (tablice ML, ale bez plików `_wazuh_suricata.json`, więc około 2,3 mln alertów Wazuh+Surika odpada) oraz `full-ait-ads.zip` (1,1 GB). Do własnego pipelineu użyj surowego zbioru i własnego parsowania — wariant “ml tables” zmienia rozkład klas. |

## 2. Stanowisko literature (podstawa, na której buduje się rozwiązanie)

| Źródło | Co mówi o AIT-ADS i triażu alertów | Co z tego wynika dla zadania |
|---|---|---|
| Landauer M. i in.: *Introducing a New Alert Data Set for Multi-Step Attack Analysis*, arXiv 2308.12627 (CSET 2023) — oficjowy opis benchmarku | 2,65 mln alertów z 3 IDS, 8 scenariuszy, 93 sygnacje detektorów; cel: prioryzacja i filtrowanie alertów, agregacja, grafy ataków. Autorzy definiują `time_label` (interwały faz z AIT-LDSv2), `event_label` (mapowanie eventów z `event_classes.ait-ads.csv` z AIT-NDS) i `detection_label` (`true_positive/false_positive/true_negative` z plików `.field.csv`), a także score detektora jako TP/(TP+FP) na interwałach ataku. Tabela 3: prioryzacja “score > 0,7” daje redukcję 56,12%, alerty w fazach ataku 1 764 581 (55,6%), SAGE 97,73%, agregacja AECID 98,93%. To jest oficjowa metryka: “reduction rate” = procent alertów, których analityk nie musi przeglądać. | To są oficjowe definicje etkiet i metryk. Moja reprodukcja `time_label` z `labels.csv` dała identyczne liczby per scenariusz (patrz §3), więc protół etkietowania jest odtwarzalny bez dostępu do AIT-LDSv2. |
| Landauer M.: *PACT: Reducing Alert Fatigue in Low-Prevalence SOC Streams with Triggered Active Learning*, arXiv 2605.22324 | AIT-ADS w ujęciu screeningowym: 587 943 alerty, 5 821 pozytywnych, prevalence 0,99%; split chronologiczny (train 100 787 / stream 487 156). Baseny: XGBoost, Random Forest, Isolation Forest. Wyniki offline: XGBoost i RF F1 95,98–96,24% przy FPR 0,035–0,037%; XGBoost-Focal w 2-fold chronologicznym: recall 76,34%, precision 98,20%, F1 84,13% (FPR 0,015%). Isolation Forest: F1 68,15±16,77% przy FPR 0,70±0,11%. Cchy meta (n-gramy, TF-IDF, one-hot, statystyki czasowe) są łączone z oryginalnymi polami; próg decisji liczony max-F1 na walidacji, nie 0,5. Autorzy wprost ostrzegają o przeciadze cech, bo część meta-cen powstaje po etkiecie lub decyzji modelu. | Najbardziej prawdopodobny punkt odniesienia “kolejnego laboratorium”: ten sam zbiór, ten sam protół (preprocessing → cechy meta → split chronologiczny → próg max-F1 → FP per mln). Moje zadanie powinno być jego uproszczoną, offlową wersją i raportować te same metryki. |
| *Let the Alerts Speak: LLM-Based IDS Alert Interpretation for SOC Triage* (Springer 2026) | Klasyfikacja `true_positive / uncertain / false_positive` z jednego alertu Surikaty: TinyBert F1 0,987, BERT-SC M 0,956, GPT-4o zero-shot F1 0,929, bez okien czasowych, z zachowaniem klasy “uncertain”; dane CVE/CWE, analiza per-plik. Model ma wspierać analityka, nie go zastępować. | Alternatywna ścieżka: LLM jako interpretator alertów (wyjaśnienie TTP), a klasyfikator tabularny jako filtr. W sprawozdaniu warto to opisać jako “dlaczego nie tylko LLM”. |
| *Context-Aware Alert Validation with NLP and LLMs* (arXiv 2509.19136), *Alert Triage 360°* (arXiv 2508.10106) | Triaż to problem kosztowy: osobno “analyst workload” i “false positive rate”; prioryzacja ma obniżać koszt triażu. | Uzasadnia raportowanie FP per 1 mln alertów i ARR obok F1. |
| Rejestr datasetów COMIDDS (AIT-ADS v1.0.1) | Formalne typy `/alert`, `/alert/ids_alert`, `/alert/audit`; przykiady z `fox_aminer.json` i `fox_wazuh.json` z polami `attack.{technicID,tacticID}`; ostrzeżenie, że pola czasu byw niespijne (np. pole `id`/`src` nie jest czasem). | Potwierdza: parsowanie czasu to osobne wyzwanie w AIT-ADS — trzeba użyć `@timestamp` (ISO) lub `DetectionTimestamp` AMinera. |
| Strona badawcza AIT-ADS <https://research.avd.ait.ac.at/projects/ait-ads/> | Zapowiada kolejne prace oceniające metodę na tym zbiorze. | Zadanie lab jest częścią szerszego protou ewalucji benchmarku. |

## 3. Moje pomiary na zbiorze (reprodukcja + analiza klas)

Metra: `research/compute_stats.py` — parser strumieniowy bez pandas (czysty stdlib), etkieta z interwałów `labels.csv` (faza ataku → `attack`, poza interwałami → `background`).

Porównanie z Tabelą 3 pracy CSET (“In attack phases” / “Filtered and in attack phases”): identyczne liczby per scenariusz (fox 421 653, harrison 431 492, russellmitchell 12 015, santos 13 004, shaw 6 935, wardbeck 7 040, wheeler 432 334, wilson 440 108) i identyczna suma 1 764 581. Wniose: protół `time_label` z `labels.csv` da się wiernie odtworzyć i nie wymaga AIT-LDSv2.

### 3.1 Rozkład klas per scenariusz

| Scenariusz | Wazuh | Suricata | AMiner | razem | attack | background | udział attack |
|---|---|---|---|---|---|---|---|
| fox | 451 439 | 11 084 | 10 581 | 473 104 | 421 653 | 51 451 | 89,1% |
| harrison | 516 370 | 67 384 | 10 194 | 593 948 | 431 492 | 162 456 | 72,6% |
| russellmitchell | 32 302 | 9 186 | 4 056 | 45 544 | 12 015 | 33 529 | 26,4% |
| santos | 82 401 | 44 112 | 4 266 | 130 779 | 13 004 | 117 775 | 9,9% |
| shaw | 51 847 | 16 692 | 2 243 | 70 782 | 6 935 | 63 847 | 9,8% |
| wardbeck | 63 694 | 24 510 | 3 053 | 91 257 | 7 040 | 84 217 | 7,7% |
| wheeler | 535 310 | 68 629 | 12 222 | 616 161 | 432 334 | 183 827 | 70,2% |
| wilson | 560 265 | 65 038 | 8 943 | 634 246 | 440 108 | 194 138 | 69,4% |
| razem | 2 293 628 | 306 635 | 55 558 | 2 655 821 | 1 764 581 | 891 240 | 66,4% |

Wniose: rzędu wielkości różnią się między scenariuszami (89,1% a 7,7%). To bezpośrednio argumentuje przeciw randomowemu splitowi — model mógłby “zapamiętywać scenariusz”.

### 3.2 Fazy ataku w klasie attack

| Faza | Alerty | Udział w klasie attack |
|---|---|---|
| dirb | 1 691 090 | 95,8% |
| wpscan | 54 931 | 3,1% |
| dnsteal | 8 630 | 0,49% |
| cracking | 5 402 | 0,31% |
| service_scans | 2 354 | 0,14% |
| network_scans | 1 579 | 0,09% |
| privilege_escalation | 371 | 0,021% |
| webshell | 121 | 0,007% |
| reverse_shell | 82 | 0,005% |
| service_stop | 21 | 0,001% |

Faza `dirb` (masowe skanowanie słów-kluszy, setki tysięcy alertów `W-Acc-400`) dominuje nad całością. Jeśli model ma sensownie działać dla faz rzadkich (privilege escalation, webshell, reverse shell, service stop), trzeba albo zrównoważyć trening (undersampling dirba / cost-weight), albo osobno mierzyć recall per faza.

### 3.3 Sygnacje detektorów

Top 60 sygnacji z mojego licznika pokrywa 2 655 315 z 2 655 821 alertów (99,98%) — reszta to długi gon ~30 sygnacji z <200 alertami (w prezentacji jest ich razem 93).

| Sygnacja | Alerty | W ataku | W tle | Udział attack |
|---|---|---|---|---|
| Wazuh `W-Acc-400` Web server 400 error code. | 1 571 313 | 1 571 057 | 256 | 0,9998 |
| Wazuh `ids` IDS event. | 295 032 | 5 766 | 289 266 | 0,0195 |
| Wazuh `W-Id-32` Dovecot Authentication Success. | 276 855 | 6 951 | 269 904 | 0,0251 |
| Wazuh `W-All-Mu4` Multiple web server 400 error codes from same source ip. | 121 794 | 121 794 | 0 | 1,0000 |
| Suricata `SURICATA TLS invalid record/traffic` | 105 043 | 2 383 | 102 660 | 0,0227 |
| Suricata `SURICATA TLS invalid handshake message` | 105 016 | 2 356 | 102 660 | 0,0227 |
| Suricata `ET INFO Observed DNS Query to .biz TLD` | 91 761 | 1 017 | 90 744 | 0,0111 |
| AMiner `A-Acc-Val1` New request method in Apache Access log. | 27 342 | 27 269 | 73 | 0,9973 |
| AMiner `A-Acc-Chr2` New characters in Apache Access request. | 10 366 | 10 051 | 315 | 0,9696 |
| AMiner `A-Id-Val1` New event type. | 7 660 | 6 286 | 1 374 | 0,8206 |
| Wazuh `W-Id-91` ClamAV database update | 8 407 | 142 | 8 265 | 0,0169 |
| Wazuh `W-Ids-61` + `W-Ids-43` Multiple IDS alerts / events | 9 398 | 154 | 9 244 | 0,0164 |
| Suricata `ET POLICY GNU/Linux APT User-Agent Outbound` | 3 422 | 37 | 3 385 | 0,0108 |

Pomiary progowe (przygotowane do sekcji o baseline’ach):

| Zbiór sygnacji | Liczba sygnacji | Alerty | W tym tła |
|---|---|---|---|
| precision ≥ 0,9 (próg ≥ 90% alertów w ataku) | 9 | 1 739 332 | 735 |
| precision ≥ 0,5 | 10 | 1 746 992 | 2 109 |
| precision < 0,5 (co najmniej 200 alertów) | 25 | 905 759 | 886 619 |
| gon < 200 alertów | 25 (w top 60) | 2 564 | — |

Trzy wnioski, które should iść do sprawozdania:

- Model, który nauczy się `rule.id` / `AnalysisComponentName`, formalnie ma basen prawie górny — dlatego baseline “prior detektora” musi być w raporcie, żeby nie chwalic modelu, który tylko odtworzył mapę reguł.
- Usunięcie 25 sygnacji z precision < 0,5 daje redukcję 34,1% całego zbioru i jednocześnie usuwa 886 619 z 891 240 alertów tła (99,5% szumu). To jest realna wartość pipelineu i punkt odniesienia do ARR z Tabeli 3 (56,12%).
- AMiner ma 4 739 na 55 558 alertów (8,5%) z `TrainingMode: true` (fox 824, harrison 539, russellmitchell 400, santos 524, shaw 408, wardbeck 750, wheeler 563, wilson 731) — to detekje z fary treningowej AMinera, które trzeba obszyć jako cecha `is_training_mode` albo wyciąć z testów.

## 4. Definicja zadania — trzy warianty i wybór

| Vignity | Cel | Etykieta | Metryki | Trudność | Rekomacja |
|---|---|---|---|---|---|
| A. Binary alert triage | filtr: alert godny uwagi vs szum | `time_label` → `attack` / `background`, 1 alert = 1 prykład | precision/recall/F1, PR-AUC, FPC, ARR, FP per 1 mln | niska do średniej | główny wynik laboratorium |
| B. Multi-class phase model | rozpoznanie fazy ataku (recon / desc / exfil / priv-esc / web / RCE + tło) | `phase_of()` z `labels.csv` (10 klas + tło, skrajny niebalans) | macro-F1, balanced accuracy, per-klasa confusion | średnia do wysokiej | opcjonalnie jako rozszerzenie |
| C. Sequential / window model | sekwencje i przewidywanie następnej fazy | okna 2 s i 2 h (jak SAGE), sekwencje alertów | PR-AUC, F1, top-k accuracy na fazach | wysoka (potrzebna agregacja, raczej GPU) | nie na to laboratorium |

Uzasadnienie wyboru A: zgodność z opisem laboratorium (“proponować model, wstępnie wytrenować system”), oficjelym celem benchmarku (“alert prioritization and filtering”), publikacją PACT (ten sam zbiór i ten sam protół) oraz tym, że variant A da się wytrenowac na CPU (16 rdzeni) w kilkanaście minut i porównac z literaturą.

## 5. Pipeline danych — konkretne kroki

```text
data/ait_ads/{scenario}_{wazuh|aminer}.json  +  data/ait_ads/labels.csv
   (1) parser strumieniowy JSONL: research/compute_stats.py → research/build_features.py
   (2) normalizacja czasu: Wazuh @timestamp (ISO) → epoch_s; AMiner DetectionTimestamp (epoch)
   (3) etkieta: interwały (scenario, start, end) → faza → binarnie attack / background
   (4) cechy wg §6
   (5) split: leave-one-scenario-out (8 foldów) + osobno split chronologiczny
   (6) trening: research/train_model.py → model/model.zip + research/model_metrics.json
   (7) raport: report/build_report.py → report/sprawozdanie.pdf
```

Dodatkowo: parser statystyczny działa bez pandas, więc można go uruchamić teraz i równolegle instalować środowisko ML (`uv pip install --system pandas pyarrow scikit-learn lightgbm joblib matplotlib`). Przy 2,7 GB danych i 12,5 GB dostępnej pamieci realnej trzeba czytać strumieniowo i zapisywać cechy do Parquet (chunki ~200 tys. linii), a nie wczytywać całego JSONL do DataFrame.

## 6. Feature engineering — lista i uzasadnienie

Cchy bez przeciążenia (dostępne przed decyzją analityka):

- `source` ∈ {wazuh, suricata, aminer} — z nagłówka `rule.description` albo `AnalysisComponentName`.
- `rule_id`, `rule_level` (skala 1-13, bardzo silny prior), `groups` (multi-hot: `web-cl`, `apache`, `ids`, `suricata`, `sqlite`, `attack_log`, `audit` dla AMinera), `decoder_name` (`web-accesslog`, `syslog`, `audit-syscall`).
- `signature_id`, `suricata.category`, `suricata.severity` dla alertów Surikiaty.
- `mitre_tactic`, `mitre_technique` — cchy priorowe, dobrze wiążą się z fazami: `TA0007` (discovery) → recon, `TA0004` (privilege escalation) → priv-esc, `TA0010` (exfiltration) → dnsteal.
- `location` (plika źródłowy, np. `/var/log/apache2/access.log`, `/var/log/auth.log`, audit systemu) i `agent.name`, `agent.ip`, `predecoder.hostname`, `program_name` — kontekst komponentu sieci.
- Pola `data.*`: `srcip`, `srcuser`, `dstip`, `dstport`, `url`, `url_query`, `http_user_agent`, `dns.query`, `name` (CVE w Surikacie). Reprezentacja: TF-IDF / character n-gramy na `url` i `full_log` oraz cechy liczbowe (długość URL, liczba segmentów, oboność `/wp-content/`, `/dirb`, licznik “wordlist-like”).
- AMiner: `frequency` z `Data.<id>.frequency`, `NotificationReport`, `ExplanationStrings` — gotowe cechy interpretacyjne.
- `is_training_mode` (AMiner `TrainingMode`), `is_aggregated_rule` (reguły agregujące typu `W-All-Mu4`, `W-Ids-61`, `W-Ids-Mu1`), licznik `firedtimes`.
- Cchy czasowe bez przeciążenia: godzina/dzień tygodnia, sekundy od startu scenariusza, “distancja do najbliższej granicy faz” (UWaga: to jest quasi-etkieta i PACT traktuje cchy czasowe jako meta-cechy; ja rekomuję ostrożnie: użyj ich tylko w wariancje z oknami 2 s/2 h, albo w ogle ich nie używaj, bo model z takiej jednej cechy potrafi odtworzyć fazę).
- Cchy agregowane, liczone strumieniowo tylko na trennu: `alert_count_60s_same_signature`, `distinct_src_in_window`, `share_of_signature_in_1min`, `rolling_rate_of_signature`.

Zabronione albo ryzykowne: `time_label` i `phase` jako cechy, `field.detection` i `field.decision` (to wprost etkieta), numeryczne `id`/`AIT:DocId`/`AIT:LogAtom*` jako identyfikatory, `full_log` surowy z datami (przecieka daty).

## 7. Protiał podziału danych

1. Leave-one-scenario-out (8 foldów) — główna ocena generalizności. Uzasadnienie: 8 scenariuszy to osobne środowiska z identycznym planem ataku (identyczny zestaw hostów, ten sam porządek faz, 23 stycznia jako data graniczna). Przy randomowym splitowi F1 potrafi wyjść ~96% zamiast ~84%, co wprost widac w wynikach PACT (offline 95,98% a chronologicznie 84,13%).
2. Chronologiczny expanding window (2 foldy, jak w PACT) — ocena stabilności w strumieniu.
3. Raport per scenariusz i per detektor — żeby widzieć, gdzie model robi FP.
4. Walidacja wewnątrz trennu do wyboru progu i hiperparametrów (osobno test LOS).

## 8. Model i trening

Poziom 0 (must have w raporcie): baseline majority class (66,4% klasy attack) i baseline “prior detektora” (mapa `rule.id → najczęstsza etkieta z trennu). To górna granica dla modeli, które nadinterpretują sygnacje.

Poziom 1 (proponowany model główny): LightGBM albo XGBoost-Focal na tabicy cech:
- `boost_from_average=false`, wagi klasowe albo `scale_pos_weight`; przy triażu kosztowym waga FP jest ważniejsza niż przewaga klasy,
- `max_depth` 8–12, `n_estimators` 300–800, `learning_rate` 0,05, early stopping na walidcji per scenariusz,
- encoding kategorialny: target / leaf-wise dla `rule_id`, `signature_id`, `location`, `agent.ip`; TF-IDF + SVD (16–64 wymiary) dla `full_log` i `url`,
- próg decisji liczony max-F1 na walidacji (nie 0,5), dodatkowo tabela progów 0,1/0,3/0,5/0,7 i wariant “FPR ≤ 0,1%”.

Poziom 2 (porównania): `HistGradientBoostingClassifier` + `class_weight='balanced'`, `LogisticRegression` na TF-IDF i cechach liczbowych jako tani, przewidywalny model, Random Forest (baseline z literatury: F1 95,98%), Isolation Forest jako sanity-check (literatura: F1 68% — unsupervised nie bije supervised przy tej prevalencji).

Poziom 3 (opcjonalnie, na następne laboracja): model sekwencyjny (LSTM/Transformer na sekwcjach met-alertów z oknami 2 s / 2 h albo attack graph mining SAGE + FlexFringe). Nie do zrobienia na CPU bez wcześniejszej agregacji — w raporcie trzeba to uargumentować jako “nie”.

Poziom 4 (LLM jako interpretator): zero-shot lub few-shot na 20–50 alertach FP (analiza wyjaśniająca, bez treningu), wg *Let the Alerts Speak*. To dobra część “dalszych badań” w sprawozdaniu.

## 9. Metryki

- PR-AUC (average precision) jako metra główna — przy 66,4% prevalence raw vs 0,99% prevalence w wariancje screeningowym PACT; ROC-AUC tylko jako uzupełnienie.
- precision / recall / F1 (binarnie i macro dla faz), confusion matrix, FPC (F1 vs false positive rate).
- FP na 1 mln alertów tła i liczba alertów do przejrzenia przez analityka — to realnie interesuje SOC.
- ARR = 1 − (alerty zachowane / 2 655 821) — porównanie z Tabelą 3 CSET: 56,12% (prioryzacja detektorów), 55,6% (okna ataku), 97,73% (SAGE), 98,93% (meta-alerty AECID). To docelowe liczby, do których laboratoryjne wyniki powinny być odniesione.
- “coverage attack phases”: recall per faza, w tym bez fary `dirb` (żeby widzieć, czy model w ogóle widzi fazy rzadkie).
- Kalbracja progu (reliability plot, Brier) oraz przełożenie na “liczbę godzin pracy analityka przy progu X”.

## 10. Format artefaktów (ZIP i PDF — sprawdzony, wykonalny układ)

`model/model.zip` — powtarzalny i samodzielny artefał:

```text
model/model.zip
├── model.joblib (sklearn pipeline) albo model.lgb / model.xgboost + model.onnx
├── encoders/  label_encoders.json, categorical_maps.json, tfidf_vectorizer.pkl
├── metadata.json  (SHA-256 danych, seed, split, parametry, lista cech, daty, metryki)
├── predict.py     (inferencja: JSONL alert → prediction; tylko pandas + lightgbm)
├── README.md      (wersje bibliotek, odtwarzanie wyniku, ograniczenia)
└── metrics.json   (LOS 8 foldów, per-scenario, per-detektor, ARR, FP per 1 mln)
```

Ryzyko: pickle i wersje bibliotek byw nieprzenośne, dlatego warto dać równolegle ONNX (`sklearn2onnx` lub `onnxruntime` dla LightGBM). Warunek laboratorium: “model w ZIP” to nie tylko plik modelu LightGBM — cały preprocessing (encodingi, TF-IDF, próg decycyjny) musi być w tym samym pliku/zipie.

`report/sprawozdanie.pdf` — układ:

1. Cel i definicja zadania, dane (SHA-256, rozmiar, liczby alertów).
2. Ekspludacja (tabela per scenariusz i per detektor; mój wynik reproducji z Tabelą 3).
3. Preprocessing i etkietowanie (`time_label`, pułapki: brak `labels.csv` w zipie, `DetectionTimestamp`, `TrainingMode`, reguły agregujące).
4. Cchy i walka z przeciążeniem (świadome decyzje).
5. Protół podziału (LOS + chronologiczny).
6. Modele i konfiguracje (tabela hiperparametrów).
7. Wyniki: krzywa PR, confusion matrix, tabela metryk, ARR, FP per 1 mln, lista 20 najlepszych FP z `full_log` i `ExplanationStrings`.
8. Analiza błędów (najczęstsze FP).
9. Ogrraniczenia (domminacja fary dirb, approxurowana etkieta, CPU, brak GPU).
10. Wnioski i dalsze badania (LLM triage, SAGE, active learning PACT).
11. Repucyjalność: repo + `run_all.sh`, czasy, seed.

Narzędzia generowania PDF: `weasyprint` albo `pandoc --pdf-engine=typst`, wykresy z matplotlib (bez wymagania LaTeX).

## 11. Ryzyka i pułapki (zapisane wprost)

| Pułka | Konsekwencja | Obsługa |
|---|---|---|
| `labels.csv` poza zipem | brak etkiety, model nie da się wytrenować | pobrane osobno z Zenodo i włożone do `data/ait_ads/labels.csv` plus hash w raporcie |
| Użycie `LogData.DetectionTimestamp` lub `predecoder.timestamp` zamiast `@timestamp` | niepoprawne okna czasowe (epoch vs ISO) | podwójna normalizacja i test reprodukcji z liczbami z Tabeli 3 |
| Dominacja fary `dirb` (95,8% klasy attack) | model uczy się “wszystko jest atakiem” | undersampling / cost-weight w trennu, osobny recall per faza rzadkie |
| Randomowy split po alertach | F1 ~96% zamiast ~84% | LOS 8 foldów i split chronologiczny, oba w raporcie |
| Reguły agregujące i duplikaty | przeciążenie trennu, zawyżone metryki | cecha `is_aggregated_rule`, ocena po skiszeniu duplikatów (`rule_id` + `srcip` + okno) |
| `TrainingMode: true` w AMiner | 4 739 alertów z fary treningowej | osobna cecha albo wycięcie z testu, opis w raporcie |
| Rozkład klas per scenariusz | model zapamiętuje scenariusz | split per scenariusz i test braku przeciążenia host/IP |
| Accuracy przy niebalansie | bezwartościowe 90% | PR-AUC, FPC, ARR |
| 12,5 GB dostępnej pamięci realnej | OOM przy pandas na 2,7 GB | parsowanie strumieniowe, zapis cech do Parquet (chunki 200 tys. linii) |

## 12. Plan implementacji

1. `uv pip install --system pandas pyarrow scikit-learn lightgbm joblib matplotlib` (CPU, 16 rdzeni).
2. `research/build_features.py` → `data/features/train_features.parquet` (chunki 200 tys. linii na bazie parsera ze statystyk).
3. `research/make_splits.py` → kolumna `fold` (LOS 8 foldów) oraz osobny podział chronologiczny.
4. `research/train_model.py` → baseline “prior detektora”, LightGBM, XGBoost-Focal, Random Forest; próg max-F1; zapis `model/model.zip` i `research/model_metrics.json`; seed 42; logi czasów.
5. `report/build_report.py` → wykresy (PR, confusion matrix, ARR per scenariusz, top 20 FP) i PDF (`weasyprint` lub `pandoc --pdf-engine=typst`).
6. `run_all.sh` w katalogu głównym z hashami danych i czasami uruchomienia — do powtarzalności w sprawozdaniu.

## 13. Źródła

1. Landauer M., Skopik F., Frank M., Hotwagner W., Wurzenberger M., Rauber A.: *Introducing a New Alert Data Set for Multi-Step Attack Analysis*, arXiv 2308.12627 / CSET 2023 — opis benchmarku, `labels.csv`, `time_label`, `event_label`, Tabela 3 z redukcją alertów.
2. Zenodo: AIT-ADS 23-01-11, <https://zenodo.org/records/8263181> (ZIP, SHA-256, `labels.csv`, `attacktimes.py`, `abbrvs.py`, `filter.py`, `analyze.py`).
3. Repo kodu autorów: <https://github.com/ait-aecid/alert-data-set> — parsowanie, priorytety detektorów, `detection_label`.
4. Landauer M. i in.: *PACT: Reducing Alert Fatigue in Low-Prevalence SOC Streams with Triggered Active Learning*, arXiv 2605.22324 — basen XGBoost-Focal na AIT-ADS (587 943 alerty, prevalence 0,99%), metryki F1/precision/FPC, split chronologiczny, meta-cechy, próg max-F1, ostrzeżenie o przeciadze cech.
5. *Let the Alerts Speak: LLM-Based IDS Alert Interpretation for SOC Triage* (Springer 2026) — TinyBert / BERT-SC / GPT-4o w triażu alertów Surikaty.
6. *Context-Aware Alert Validation with NLP and LLMs* (arXiv 2509.19136), *Alert Triage 360°* (arXiv 2508.10106) — triaż jako problem kosztowy.
7. Rejestr datasetów COMIDDS (AIT-ADS v1.0.1, typy `/alert`, `/alert/ids_alert`, `/alert/audit`) i strona projektu <https://research.avd.ait.ac.at/projects/ait-ads/>.
8. Prezentacja laboratoryjna `AIT_ADS_Wazuh_wyjasnienie (1).pptx` — liczby 2 293 628 / 306 635 / 55 558 / 2 655 821, scenariusze i fary ataku, 23 stycznia jako data graniczna dla kolejnych laborów.
