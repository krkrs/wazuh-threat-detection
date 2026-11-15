# Wazuh Threat Detection — instrukcja uruchomienia i odtworzenia treningu

## 1. Cel projektu

Prototyp klasyfikacji faz ataków na podstawie **alertów Wazuh** ze zbioru AIT-ADS.
Nie używamy alertów Suricata ani AMiner jako danych wejściowych modelu.

Model GRU analizuje sześć kolejnych 10-sekundowych okien (łącznie 60 s),
po 27 cech w każdym. Klasy: `background`, `cracking`, `dirb`, `dnsteal`,
`network_scans`, `privilege_escalation`, `reverse_shell`, `service_scans`,
`webshell`, `wpscan`. Faza `service_stop` nie jest rozpoznawana, ponieważ
w wykorzystanych oknach nie było dla niej przykładów uczących.

**To prototyp badawczy, nie gotowy produkcyjny system wykrywania włamań.**
Wynik `predicted_score` pochodzi z softmax i nie jest skalibrowanym
prawdopodobieństwem rzeczywistego ataku.

## 2. Minimalny zestaw plików do uruchomienia API

```text
wazuh-threat-detection/
├── README.md
├── api.py                       # FastAPI z walidacją i limitami żądań
├── wazuh_predictor.py           # GRU + konwersja alertów + budowanie sekwencji
└── models/
    └── wazuh_gru_multiclass.pt  # zweryfikowany checkpoint referencyjny
```

**Ważne:** `api.py`, `wazuh_predictor.py` i `.pt` trzeba skopiować
z działającego projektu. Sama dokumentacja ani notatnik treningowy
nie zastępują tych plików.

API nie wymaga pobrania dwóch milionów alertów, pliku `labels.csv`,
Jupytera ani ponownego treningu.

## 3. Uruchomienie API w Windows PowerShell

Instrukcja sprawdzona w Windows, Python 3.13.18 i `uv`. Polecenia
wykonuj w głównym katalogu projektu. **Nie przenoś checkpointu**:
`api.py` szuka `models/wazuh_gru_multiclass.pt` względem własnego katalogu.

Do uruchomienia API służy `requirements.txt`. Plik
`requirements-training.txt` dodaje biblioteki potrzebne do Jupytera.
Nie kopiuj folderów `.venv*` od innego członka zespołu.

```powershell
# 1. Utwórz nowe środowisko, jeśli jeszcze nie istnieje.
uv venv .venv-gru --python 3.13

# 2. PyTorch CPU z właściwego indeksu — ważne dla wersji 2.15.0+cpu.
uv pip install --python .venv-gru\Scripts\python.exe "torch==2.15.0+cpu" --index https://download.pytorch.org/whl/cpu

# 3. Zainstaluj pozostałe biblioteki.
uv pip install --python .venv-gru\Scripts\python.exe -r requirements.txt

# 4. Uruchom usługę lokalnie.
.venv-gru\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

**Dlaczego instalacja jest dwuetapowa?** Samo `uv pip install -r
requirements.txt` w czystym środowisku nie znalazło wersji
`torch==2.15.0+cpu`. Wskazanie indeksu PyTorch w pierwszym kroku
rozwiązało ten problem. Po instalacji zgłoszono `IMPORTY OK`
i `PyTorch: 2.15.0+cpu`, a nowe środowisko uruchomiło API z odpowiedzią
`GET /health` — HTTP 200.

Sprawdź:
- `http://127.0.0.1:8000/health` — stan usługi i lista klas;
- `http://127.0.0.1:8000/docs` — interaktywna dokumentacja HTTP.

API działa lokalnie na `127.0.0.1`. Nie udostępniaj go publicznie:
nie ma uwierzytelniania. Ma prototypowe ograniczenia: maksymalnie
20 000 alertów i 20 MiB treści żądania. Poprawne małe żądania
i odrzucanie deklarowanego nadmiarowego rozmiaru przeszły testy;
pełne testy transferu dużych danych strumieniowych nie są zakończone.

## 4. Jak wysłać predykcję

`POST http://127.0.0.1:8000/predict`

Żądanie JSON ma cztery pola:

- `alerts` — wszystkie dostępne alerty w historii 60 sekund,
  każdy z polami `@timestamp`, `rule.id`, `rule.level`;
- `target_start` — początek **ostatniego zamkniętego okna** 10 s,
  wyrównany do 10 s, ze strefą czasową;
- `history_complete` — `true` wyłącznie wtedy, gdy klient
  rzeczywiście dostarczył pełną historię;
- `scenario` — opcjonalny identyfikator strumienia; **nie jest cechą modelu**.

Przykład **formatu** (zakłada, że w podanym okresie naprawdę
nie było innych alertów; serwer nie może tego sprawdzić):

```json
{
  "alerts": [
    {
      "@timestamp": "2026-11-15T12:00:01Z",
      "rule": {"id": "9701", "level": 3},
      "decoder": {"name": "dovecot"}
    }
  ],
  "target_start": "2026-11-15T12:00:00Z",
  "history_complete": true,
  "scenario": "demo"
}
```

Serwer zwraca przewidywaną fazę `predicted_phase`, wynik softmax
`predicted_score`, `scores` dla 10 klas, informację o deklarowanej
kompletności historii i początek okna. Niepełne/niepoprawne żądania
otrzymują kod HTTP 422; obecnie nie klasyfikujemy okien docelowych
bez żadnych alertów.

Serwer nie pobiera jeszcze alertów automatycznie z działającego
Wazuh Managera: dane trzeba wysłać do API.

## 5. Odtworzenie treningu — tylko dla osób rozwijających model

Dodatkowe pliki:

```text
wazuh_trening_gru_poprawiony.ipynb  # w katalogu głównym; zachowuje działające ścieżki
data/
└── ait_ads/
    ├── labels.csv
    ├── fox_wazuh.json
    ├── harrison_wazuh.json
    └── ...  # pozostałe sześć plików *_wazuh.json
```

Do odtworzenia treningu użyj dodatkowo `requirements-training.txt`
(po instalacji PyTorch CPU, jak w sekcji 3). Uruchom notatnik
po kolei w środowisku Jupyter. W razie potrzeby dostosuj `DATA_DIR` w pierwszej
komórce do faktycznej lokalizacji danych.

Notatnik:
1. Filtruje dane do alertów Wazuh i nadaje etykiety na podstawie
   przedziałów czasowych z `labels.csv`.
2. Agreguje alerty w okna 10 s i usuwa okna graniczne.
3. Buduje sekwencje sześciu okien i 27 cech wejściowych.
4. Trenuje GRU na siedmiu scenariuszach, pozostawiając Harrison
   poza treningiem.
5. Zapisuje `models/wazuh_gru_reproduced.pt`, **bez nadpisywania**
   `wazuh_gru_multiclass.pt`.

Weryfikowane wyniki przygotowania danych:
- 1 986 905 alertów po filtracji;
- 63 262 okna zawierające alerty, 30 odrzuconych, 63 232 końcowe;
- 55 672 okna w treningu końcowym, 7560 odłożonych dla Harrisona;
- macierz sekwencji `(63232, 6, 27)`;
- 12 epok, końcowa strata treningowa `0.5025`.

**Odtwarzalność potwierdzona 15.11.2026:** uruchomienie
uporządkowanego notatnika dało checkpoint o identycznych wagach
jak model referencyjny (maksymalna różnica wag `0.0`,
ta sama kolejność cech i klas).

## 6. Wyniki i ograniczenia

W jednorazowym końcowym teście na odłożonym Harrisonie
(7560 okien, 562 oznaczone jako faza ataku):
- dokładność `0.9481`;
- macro F1 dla dziesięciu klas `0.4872`;
- recall oznaczonych okien ataku `0.3043`;
- recall poprawnie rozpoznanego typu `0.3025`;
- 0 fałszywych alarmów w 6998 oknach oznaczonych jako tło;
- na poziomie epizodów: 6/9 wykrytych, 5/9 z poprawnym typem
  (dla epizodów mających przynajmniej jedno pełne okno).

Model słabo wykrywa część faz, zwłaszcza `cracking`, `dnsteal`
i `network_scans`; `service_stop` nie jest klasą wyjściową.
Etykiety dotyczą **przedziałów czasowych ataków**, a nie zawsze
bezspornie złośliwych pojedynczych alertów.

## 7. Archiwum eksperymentów

Oryginalny `wazuh_etap1_audyt(1).ipynb` należy zachować w `archiwum/`:
audyt, modele RF/GRU/LSTM, walidacje między scenariuszami,
diagnostykę danych i testy integracyjne. Nie jest to instrukcja
uruchamiania aplikacji i nie trzeba go wykonywać podczas pracy z API.
