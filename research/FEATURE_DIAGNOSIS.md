# Feature collection audit (dropped features)

Audited `artifacts/feature_plan.json` -> `dropped_features` (34 columns) against the
raw Wazuh stream `data/ait_ads/*_wazuh.json` (2,600,263 rows) and the feature tables
(2,293,628 rows). The audit answers one question: is a drop a data-collection failure
or a correct decision?

## Method

- Per-column non-null / non-zero counts in `data/features/*.features.parquet`, per scenario.
- Raw-source availability probe: streaming JSON parse counting source-key presence
  (`predecoder.program_name`, `rule.mitre_*`, `data.dstport`, `data.syscall`,
  `data.database_name`, `data.rule_md5`, `data.sha256/md5`, `data.http`, `data.url`),
  plus text probes in `full_log` (UA tokens, URL tokens, `query:`, DNS request shapes).
- Phase/label attribution of raw hits via `labels.csv` intervals.

## Verdict per family

| family | dropped cols | raw availability | verdict |
|---|---|---|---|
| URL path flags | `url_semicolon`, `url_traversal`, `urlflag_env`, `urlflag_json` | `;` appears 1,669,630x in `full_log` (referrer/UA text), never in decoded `data.url`; `../` / `.env` / `.json` absent | correct: dataset has no traversal/env payloads; pipeline uses decoded `data.url` (1,696,191 rows), so no info loss |
| DNS family | `dns_len`, `dns_labels`, `dns_max_label_len`, `dns_avg_label_len`, `dns_digits`, `dns_letters`, `dns_digit_ratio`, `dns_entropy` | `query:` 0 hits, DNS request regex 0 hits, `dnsteal` location 0 rows | correct: DNS payloads live in Suricata/AMiner streams (removed); Wazuh `dnsteal` rows are Dovecot/Freshclam syslog only (shaw: 105 rows, decoder dovecot) |
| File integrity / MITRE / audit | `sha256_present`, `rule_md5`, `syscall`, `db_name`, `mitre_tactic`, `mitre_technique` | `data.sha256/md5` 0, `data.rule_md5` 0, `data.syscall` 0, `data.database_name` 0, `rule.mitre_*` 0 | correct: keys do not exist in this Wazuh export |
| Network port | `dst_port` | `data.dstport`/`dest_port` 306,635 rows - all in `suricata/eve.json` | correct after Suricata removal (Wazuh stream has no port fields) |
| Program name | `program_name` | `predecoder.program_name` 287,911 rows (11%) | intentional: host-identity suppression (`build_features.py:368-370` commented out) - leakage guard, not a bug |
| Tool UA flags | `ua_nikto`, `ua_sqlmap`, `ua_curl`, `ua_python`, `ua_golang`, `ua_apt`, `ua_headless`, `ua_java`, `ua_openvas`, `ua_masscan`, `ua_zmap`, `ua_scrapy`, `ua_apachebench`, `ua_persistent` | token scan in `full_log`/`http_user_agent`: dirb 740, wpscan 27,269, nmap 318, ncat 371, wget 362, HeadlessChrome 389; APT-HTTP 3,422 - **all 3,422 are Suricata `eve.json` rows** (`rule.groups=[ids,suricata]`, excluded from the Wazuh-only stream); nikto/sqlmap/curl/python-requests/OpenVAS/masscan/zmap/Java/Scrapy/ApacheBench 0 hits | **BUG (fixed): case mismatch** - `UA_PATTERNS` needles are CamelCase (`APT-HTTP`, `HeadlessChrome`, `Go-http-client`, `Java/`, `OpenVAS`, `Scrapy`, `ApacheBench`) while the scan text was lowercased -> needles never matched. `ua_headless` (389 rows, web-automation signal) was silently zero; `ua_apt` fires only on Suricata rows, so it is legitimately 0 in the Wazuh-only stream. Rest of the family: legitimately absent (0 rows) |

## Fix applied

`research/build_features.py` -> `url_features`: UA needle comparison is now case-insensitive

```python
    low_scan = (scan_text or "").lower()
    for flag, needle in UA_PATTERNS.items():
        # needles are case-insensitive here; low_scan is already lowercased
        out[flag] = float(needle.lower() in low_scan)
```

Effect after rebuild (measured, no training): `ua_headless` enters the plan, `ua_apt` stays 0 (Suricata-only),
numeric 55 -> 56, dropped 34 -> 33 (`artifacts/feature_plan_ua_fix.json`).
UA feature counts in the rebuilt 2,293,628-row table:
`ua_dirb` 740, `ua_wpscan` 27,269, `ua_nmap` 318 (315 attack), `ua_ncat` 371 (all attack),
`ua_wget` 362 (all attack), `ua_headless` 389 (382 background, 7 attack - cracking 1, dirb 2, dnsteal 4).
HeadlessChrome 389 rows (web automation), Nmap UA in UA-block rows: service_scans 138 rows
(russellmitchell 27, wilson 27, fox 24, wheeler 24, santos 15, harrison 12, wardbeck 9).

## Notes for the report

- DNS/URL/audit constants: structural absence in Wazuh-only stream (feature plan drop is correct,
  not a collection bug); the report's DNS claim is a 2x34 zero-fill bug, not a data gap.
- `dst_port`, `syscall`, `db_name`, `rule_md5`, `sha256_present`, MITRE columns: drop is correct.
- `program_name`: intentional host-identity suppression, documented in code.
- UA flags: case-sensitivity bug fixed; `ua_apt`/`ua_headless` will be available after rebuild,
  all other UA flags stay zero because the dataset never uses those tools through Wazuh-visible HTTP.
