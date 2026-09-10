# SCHEMAS.md — контракты между слоями

**Сфера:** JSON-схемы всех стыков конвейера.
**Обновлено:** 2026-09-03.
**Диалект:** JSON Schema 2020-12.

Единственный файл в проекте, на который не распространяется лимит в 200 строк:
его объём задаётся числом контрактов, а не изложением.

Правила:
- Каждый слой валидирует свой вход и свой выход. Ошибка валидации — исключение, не warning.
- Все временные метки — ISO-8601 UTC с суффиксом `Z`.
- Все денежные величины — `number`, доллары США.
- `additionalProperties: false` везде, где схема описывает выход модели.

---

## Общие определения

```json
{
  "$id": "scout/defs",
  "$defs": {
    "Provenance": {
      "type": "object",
      "required": ["path", "commit_sha", "retrieved_at", "api"],
      "additionalProperties": false,
      "properties": {
        "path": { "type": "string", "description": "путь в репозитории или 'metadata'" },
        "commit_sha": { "type": "string", "pattern": "^[0-9a-f]{7,40}$" },
        "retrieved_at": { "type": "string", "format": "date-time" },
        "api": { "enum": ["search", "repos", "contents", "trees", "licenses"] }
      }
    },
    "TokenUsage": {
      "type": "object",
      "required": ["model", "input_tokens", "output_tokens", "cost_usd", "pricing_window"],
      "additionalProperties": false,
      "properties": {
        "model": { "enum": ["deepseek-v4-flash", "deepseek-v4-pro", "qwen3.8-max"] },
        "input_tokens": { "type": "integer", "minimum": 0 },
        "cached_input_tokens": { "type": "integer", "minimum": 0, "default": 0 },
        "output_tokens": { "type": "integer", "minimum": 0 },
        "cost_usd": { "type": "number", "minimum": 0 },
        "pricing_window": { "enum": ["off-peak", "peak"] }
      }
    },
    "EffortDays": {
      "type": "object",
      "description": "трёхточечная оценка трудозатрат на интеграцию",
      "required": ["low", "likely", "high"],
      "additionalProperties": false,
      "properties": {
        "low": { "type": "number", "minimum": 0 },
        "likely": { "type": "number", "minimum": 0 },
        "high": { "type": "number", "minimum": 0 }
      }
    }
  }
}
```

---

## 1. `ScanRequest` — вход конвейера

```json
{
  "$id": "scout/scan_request",
  "type": "object",
  "required": ["request_id", "query_text", "created_at", "options"],
  "additionalProperties": false,
  "properties": {
    "request_id": { "type": "string", "format": "uuid" },
    "query_text": { "type": "string", "minLength": 8, "maxLength": 500 },
    "created_at": { "type": "string", "format": "date-time" },
    "options": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "max_candidates": { "type": "integer", "minimum": 5, "maximum": 50, "default": 50 },
        "audit_limit":    { "type": "integer", "minimum": 1, "maximum": 10, "default": 10 },
        "report_limit":   { "type": "integer", "minimum": 1, "maximum": 5,  "default": 5 },
        "off_peak":       { "type": "boolean", "default": false },
        "refresh":        { "type": "boolean", "default": false }
      }
    }
  }
}
```

Пример:

```json
{
  "request_id": "0f6d1a2e-8c4b-4f1a-9d3e-1b2c3d4e5f60",
  "query_text": "нужен парсер PDF-таблиц на Python с сохранением структуры",
  "created_at": "2026-09-03T12:00:00Z",
  "options": { "max_candidates": 50, "audit_limit": 10, "report_limit": 5,
               "off_peak": false, "refresh": false }
}
```

---

## 2. `Intent` — выход извлечения интента (V4-Flash)

```json
{
  "$id": "scout/intent",
  "type": "object",
  "required": ["request_id", "task", "domain", "languages", "must_have",
               "exclude", "synonyms", "known_libraries", "model", "prompt_version"],
  "additionalProperties": false,
  "properties": {
    "request_id": { "type": "string", "format": "uuid" },
    "task": { "type": "string", "maxLength": 200,
              "description": "переформулировка задачи одним предложением" },
    "domain": { "type": "array", "items": { "type": "string" }, "maxItems": 5 },
    "languages": { "type": "array", "items": { "type": "string" }, "maxItems": 3,
                   "description": "пусто = язык не важен" },
    "must_have": { "type": "array", "items": { "type": "string" }, "maxItems": 6 },
    "nice_to_have": { "type": "array", "items": { "type": "string" }, "maxItems": 6 },
    "exclude": { "type": "array", "items": { "type": "string" }, "maxItems": 6 },
    "synonyms": { "type": "array", "items": { "type": "string" }, "minItems": 3, "maxItems": 10,
                  "description": "англоязычные формулировки задачи" },
    "known_libraries": { "type": "array", "items": { "type": "string" }, "maxItems": 8,
                         "description": "гипотезы об именах известных библиотек" },
    "model": { "const": "deepseek-v4-flash" },
    "prompt_version": { "type": "string", "pattern": "^intent-\\d+$" }
  }
}
```

Пример:

```json
{
  "request_id": "0f6d1a2e-8c4b-4f1a-9d3e-1b2c3d4e5f60",
  "task": "извлечение таблиц из PDF в структурированный вид",
  "domain": ["pdf", "data-extraction", "tables"],
  "languages": ["python"],
  "must_have": ["сохранение строк и столбцов", "работа без облачного API"],
  "nice_to_have": ["CLI", "экспорт в CSV/DataFrame"],
  "exclude": ["платные SaaS-обёртки", "только-OCR решения"],
  "synonyms": ["pdf table extraction", "extract tables from pdf",
               "pdf table parser", "tabular data from pdf", "pdf to csv tables"],
  "known_libraries": ["camelot", "tabula-py", "pdfplumber", "unstructured"],
  "model": "deepseek-v4-flash",
  "prompt_version": "intent-1"
}
```

---

## 3. `SearchQuerySet` — выход генератора запросов

```json
{
  "$id": "scout/search_query_set",
  "type": "object",
  "required": ["request_id", "queries", "generated_at", "generator_version"],
  "additionalProperties": false,
  "properties": {
    "request_id": { "type": "string", "format": "uuid" },
    "generated_at": { "type": "string", "format": "date-time" },
    "generator_version": { "type": "string", "pattern": "^qg-\\d+$" },
    "queries": {
      "type": "array", "minItems": 5, "maxItems": 10,
      "items": {
        "type": "object",
        "required": ["id", "family", "q", "sort", "per_page"],
        "additionalProperties": false,
        "properties": {
          "id": { "type": "string", "pattern": "^q\\d+$" },
          "family": { "enum": ["exact", "synonym", "topic", "library",
                               "broad", "readme", "recent"] },
          "q": { "type": "string", "maxLength": 256,
                 "description": "строка запроса GitHub Search с квалификаторами" },
          "sort": { "enum": ["stars", "updated", "best-match"] },
          "per_page": { "type": "integer", "minimum": 10, "maximum": 30 }
        }
      }
    }
  }
}
```

Пример одного элемента:

```json
{ "id": "q3", "family": "topic",
  "q": "topic:pdf topic:table-extraction archived:false",
  "sort": "stars", "per_page": 30 }
```

Семантика семейств — в `QUERIES.md`.

---

## 4. `Candidate` — вход Слоя 1 (после дедупа и RRF)

```json
{
  "$id": "scout/candidate",
  "type": "object",
  "required": ["repo_id", "full_name", "html_url", "stars", "archived", "is_fork",
               "pushed_at", "default_branch", "head_sha", "found_by", "rrf_score",
               "rank", "retrieved_at"],
  "additionalProperties": false,
  "properties": {
    "repo_id": { "type": "integer" },
    "full_name": { "type": "string", "pattern": "^[^/]+/[^/]+$" },
    "html_url": { "type": "string", "format": "uri" },
    "description": { "type": ["string", "null"], "maxLength": 500 },
    "language": { "type": ["string", "null"] },
    "topics": { "type": "array", "items": { "type": "string" }, "maxItems": 20 },
    "stars": { "type": "integer", "minimum": 0 },
    "forks": { "type": "integer", "minimum": 0 },
    "open_issues": { "type": "integer", "minimum": 0 },
    "archived": { "type": "boolean" },
    "is_fork": { "type": "boolean" },
    "created_at": { "type": "string", "format": "date-time" },
    "pushed_at": { "type": "string", "format": "date-time" },
    "default_branch": { "type": "string" },
    "head_sha": { "type": "string", "pattern": "^[0-9a-f]{7,40}$" },
    "license_spdx": { "type": ["string", "null"],
                      "description": "из GitHub API; уточняется на Слое 2" },
    "found_by": { "type": "array", "items": { "type": "string", "pattern": "^q\\d+$" },
                  "minItems": 1 },
    "rrf_score": { "type": "number", "minimum": 0 },
    "prior_score": { "type": "number", "minimum": 0, "maximum": 1 },
    "rank": { "type": "integer", "minimum": 1 },
    "retrieved_at": { "type": "string", "format": "date-time" }
  }
}
```

---

## 5. `ScreeningResult` — выход Слоя 1 (V4-Flash)

```json
{
  "$id": "scout/screening_result",
  "type": "object",
  "required": ["request_id", "layer", "model", "prompt_version", "results",
               "passed", "token_usage"],
  "additionalProperties": false,
  "properties": {
    "request_id": { "type": "string", "format": "uuid" },
    "layer": { "const": 1 },
    "model": { "const": "deepseek-v4-flash" },
    "prompt_version": { "type": "string", "pattern": "^l1-\\d+$" },
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["repo_id", "full_name", "relevance", "verdict", "reasons"],
        "additionalProperties": false,
        "properties": {
          "repo_id": { "type": "integer" },
          "full_name": { "type": "string" },
          "relevance": { "type": "number", "minimum": 0, "maximum": 1 },
          "verdict": { "enum": ["pass", "reject"] },
          "reasons": { "type": "array", "items": { "type": "string", "maxLength": 200 },
                       "minItems": 1, "maxItems": 3 },
          "red_flags": {
            "type": "array",
            "items": { "enum": ["demo-or-tutorial", "wrong-domain", "wrong-language",
                                "saas-wrapper", "abandoned", "no-code", "duplicate-of-known"] }
          },
          "evidence": { "type": "array", "items": { "$ref": "scout/defs#/$defs/Provenance" } }
        }
      }
    },
    "passed": { "type": "array", "items": { "type": "integer" }, "maxItems": 10,
                "description": "repo_id, отсортированные по relevance убыв." },
    "token_usage": { "$ref": "scout/defs#/$defs/TokenUsage" }
  }
}
```

Запрет: `reasons` — проза на русском, без цитат из README длиннее 120 символов.

---

## 6. `LicensePassport`

```json
{
  "$id": "scout/license_passport",
  "type": "object",
  "required": ["spdx_id", "detected_by", "confidence", "copyleft",
               "code_reuse_allowed", "obligations", "source"],
  "additionalProperties": false,
  "properties": {
    "spdx_id": { "type": ["string", "null"], "description": "null = лицензия не определена" },
    "name": { "type": ["string", "null"] },
    "detected_by": { "enum": ["github-api", "license-file", "readme-mention", "none"] },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
    "copyleft": { "enum": ["none", "weak", "strong", "unknown"] },
    "network_copyleft": { "type": "boolean", "description": "true для AGPL" },
    "commercial_use": { "type": ["boolean", "null"] },
    "attribution_required": { "type": ["boolean", "null"] },
    "share_alike": { "type": ["boolean", "null"] },
    "code_reuse_allowed": { "type": "boolean",
      "description": "false при copyleft strong/unknown — тогда только реинжиниринг" },
    "obligations": { "type": "array", "items": { "type": "string", "maxLength": 200 } },
    "source": { "$ref": "scout/defs#/$defs/Provenance" }
  }
}
```

Если `spdx_id: null` или `copyleft: "unknown"`, то `code_reuse_allowed` обязан быть `false`,
а вердикт кандидата не может быть `USE`.

---

## 7. `AuditResult` — выход Слоя 2 (V4-Pro), единица кэширования

```json
{
  "$id": "scout/audit_result",
  "type": "object",
  "required": ["repo_id", "full_name", "head_sha", "audited_at", "model",
               "prompt_version", "structure", "dependencies", "license_passport",
               "maintenance", "fit", "risks", "score", "verdict", "provenance"],
  "additionalProperties": false,
  "properties": {
    "repo_id": { "type": "integer" },
    "full_name": { "type": "string" },
    "head_sha": { "type": "string", "pattern": "^[0-9a-f]{7,40}$" },
    "audited_at": { "type": "string", "format": "date-time" },
    "model": { "enum": ["deepseek-v4-pro", "qwen3.8-max"] },
    "prompt_version": { "type": "string", "pattern": "^l2-\\d+$" },

    "structure": {
      "type": "object",
      "additionalProperties": false,
      "required": ["entrypoints", "modules", "has_tests", "has_ci", "has_docs"],
      "properties": {
        "entrypoints": { "type": "array", "items": { "type": "string" }, "maxItems": 10 },
        "modules": { "type": "array", "items": { "type": "string" }, "maxItems": 30 },
        "has_tests": { "type": "boolean" },
        "test_paths": { "type": "array", "items": { "type": "string" }, "maxItems": 10 },
        "has_ci": { "type": "boolean" },
        "ci_files": { "type": "array", "items": { "type": "string" }, "maxItems": 10 },
        "has_docs": { "type": "boolean" },
        "file_count": { "type": "integer", "minimum": 0 }
      }
    },

    "dependencies": {
      "type": "object",
      "additionalProperties": false,
      "required": ["manifest", "runtime", "count"],
      "properties": {
        "manifest": { "type": ["string", "null"] },
        "runtime": {
          "type": "array", "maxItems": 40,
          "items": {
            "type": "object",
            "required": ["name"],
            "additionalProperties": false,
            "properties": {
              "name": { "type": "string" },
              "constraint": { "type": ["string", "null"] }
            }
          }
        },
        "count": { "type": "integer", "minimum": 0 },
        "heavy": { "type": "array", "items": { "type": "string" }, "maxItems": 10,
                   "description": "зависимости с тяжёлой установкой или нативными сборками" }
      }
    },

    "license_passport": { "$ref": "scout/license_passport" },

    "maintenance": {
      "type": "object",
      "additionalProperties": false,
      "required": ["last_commit", "open_issues"],
      "properties": {
        "last_commit": { "type": "string", "format": "date-time" },
        "commits_90d": { "type": ["integer", "null"], "minimum": 0 },
        "contributors_12m": { "type": ["integer", "null"], "minimum": 0 },
        "open_issues": { "type": "integer", "minimum": 0 },
        "releases_12m": { "type": ["integer", "null"], "minimum": 0 }
      }
    },

    "fit": {
      "type": "object",
      "additionalProperties": false,
      "required": ["covers", "gaps", "integration_effort_days"],
      "properties": {
        "covers": { "type": "array", "items": { "type": "string", "maxLength": 200 } },
        "gaps":   { "type": "array", "items": { "type": "string", "maxLength": 200 } },
        "integration_effort_days": { "$ref": "scout/defs#/$defs/EffortDays" }
      }
    },

    "risks": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["type", "severity", "note"],
        "additionalProperties": false,
        "properties": {
          "type": { "enum": ["license", "maintenance", "dependency",
                             "api-stability", "scope-mismatch", "single-maintainer"] },
          "severity": { "enum": ["low", "medium", "high"] },
          "note": { "type": "string", "maxLength": 300 }
        }
      }
    },

    "score": {
      "type": "object",
      "additionalProperties": false,
      "required": ["relevance", "quality", "maintenance", "license", "total"],
      "properties": {
        "relevance":   { "type": "number", "minimum": 0, "maximum": 1 },
        "quality":     { "type": "number", "minimum": 0, "maximum": 1 },
        "maintenance": { "type": "number", "minimum": 0, "maximum": 1 },
        "license":     { "type": "number", "minimum": 0, "maximum": 1 },
        "total":       { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },

    "verdict": { "enum": ["USE", "FORK", "BUILD"] },
    "verdict_rationale": { "type": "string", "maxLength": 600 },
    "provenance": { "type": "array", "items": { "$ref": "scout/defs#/$defs/Provenance" },
                    "minItems": 1 },
    "token_usage": { "$ref": "scout/defs#/$defs/TokenUsage" }
  }
}
```

`total` считается детерминированным кодом, не моделью:
`total = 0.40*relevance + 0.25*quality + 0.20*maintenance + 0.15*license`.

Правила вердикта (применяются кодом после получения оценок):
- `USE` — `total ≥ 0.70`, `gaps` пуст или тривиален, `code_reuse_allowed: true`.
- `FORK` — `total ≥ 0.50` и есть `gaps`, либо `copyleft: weak`.
- `BUILD` — во всех остальных случаях, включая `copyleft: strong` и `spdx_id: null`.

> **Примечание:** 0.70/0.50 здесь — пороги вердикта; они не связаны с весами
> RRF 0.70/0.30 из `QUERIES.md`. Совпадение чисел случайное, «унификация»
> сломает обе формулы.

---

## 8. `Report` — выход конвейера

```json
{
  "$id": "scout/report",
  "type": "object",
  "required": ["request_id", "query_text", "generated_at", "mode",
               "recommendation", "candidates", "cost_usd", "duration_sec", "cache"],
  "additionalProperties": false,
  "properties": {
    "request_id": { "type": "string", "format": "uuid" },
    "query_text": { "type": "string" },
    "generated_at": { "type": "string", "format": "date-time" },
    "mode": { "enum": ["sync", "off-peak"] },
    "recommendation": { "enum": ["USE", "FORK", "BUILD"] },
    "recommendation_target": { "type": ["string", "null"],
                               "description": "full_name или null для BUILD" },
    "rationale": { "type": "string", "maxLength": 1000 },
    "candidates": {
      "type": "array", "maxItems": 5,
      "items": {
        "type": "object",
        "required": ["rank", "full_name", "html_url", "verdict", "score",
                     "strengths", "weaknesses", "license_passport",
                     "integration_effort_days", "provenance"],
        "additionalProperties": false,
        "properties": {
          "rank": { "type": "integer", "minimum": 1, "maximum": 5 },
          "full_name": { "type": "string" },
          "html_url": { "type": "string", "format": "uri" },
          "head_sha": { "type": "string" },
          "verdict": { "enum": ["USE", "FORK", "BUILD"] },
          "score": { "type": "number", "minimum": 0, "maximum": 1 },
          "strengths": { "type": "array", "items": { "type": "string", "maxLength": 200 },
                         "minItems": 1, "maxItems": 5 },
          "weaknesses": { "type": "array", "items": { "type": "string", "maxLength": 200 },
                          "maxItems": 5 },
          "license_passport": { "$ref": "scout/license_passport" },
          "integration_effort_days": { "$ref": "scout/defs#/$defs/EffortDays" },
          "provenance": { "type": "array", "items": { "$ref": "scout/defs#/$defs/Provenance" } }
        }
      }
    },
    "partial": {
      "type": "boolean", "default": false,
      "description": "часть данных потеряна из-за сбоя: невыполненная выдача, отказ GitHub, кандидат, чей ответ не разобран"
    },
    "dropped": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["full_name", "stage", "reason"],
        "additionalProperties": false,
        "properties": {
          "full_name": { "type": "string" },
          "stage": { "enum": ["search", "screening", "audit"] },
          "reason": { "type": "string", "maxLength": 200 }
        }
      }
    },
    "queries_used": { "type": "array", "items": { "type": "string" },
                      "description": "строки запросов — для отладки и воспроизводимости" },
    "cost_usd": { "type": "number", "minimum": 0 },
    "duration_sec": { "type": "number", "minimum": 0 },
    "cache": {
      "type": "object",
      "additionalProperties": false,
      "required": ["audits_hit", "audits_miss"],
      "properties": {
        "audits_hit": { "type": "integer", "minimum": 0 },
        "audits_miss": { "type": "integer", "minimum": 0 }
      }
    }
  }
}
```

Отчёт сериализуется в JSON (машинный контракт) и рендерится в Markdown для человека.
Markdown — производная от JSON, а не отдельный источник истины.

`partial` и `dropped` описывают разное. `dropped` — штатное выбытие кандидата:
репозиторий удалён или стал приватным между поиском и аудитом, выдача не сложилась
в `Candidate`. Такой отчёт полон. `partial` поднимается только когда данные потеряны
из-за сбоя — невыполненная выдача, отказ GitHub на дозапросе, кандидат, чей ответ
модель дважды не смогла отдать по схеме. Выбывший кандидат попадает в `dropped`
в обоих случаях, но `partial` — лишь во втором.

---

## 9. `CacheEntry` — строка в SQLite

```json
{
  "$id": "scout/cache_entry",
  "type": "object",
  "required": ["key", "repo_id", "head_sha", "payload_type", "payload", "created_at"],
  "additionalProperties": false,
  "properties": {
    "key": { "type": "string", "pattern": "^repo:\\d+:[0-9a-f]{7,40}:l2-\\d+$" },
    "repo_id": { "type": "integer" },
    "head_sha": { "type": "string" },
    "prompt_version": { "type": "string", "pattern": "^l2-\\d+$" },
    "payload_type": { "const": "audit_result_v1" },
    "payload": { "$ref": "scout/audit_result" },
    "created_at": { "type": "string", "format": "date-time" },
    "hits": { "type": "integer", "minimum": 0, "default": 0 }
  }
}
```

Инвалидация по двум осям: смена `head_sha` (изменился код) и смена
`prompt_version` (изменился вопрос, который мы коду задаём). Никакого TTL по времени.

**Почему версия промпта в ключе.** `AuditResult` — это не свойство репозитория,
а ответ конкретного промпта о репозитории. Без версии в ключе бамп до `l2-2`
молча отдавал бы суждения `l2-1`, и замер «до и после», которого требует
`CHECKLIST.md` при любой правке промпта, показывал бы «до» оба раза. Ошибка
при этом тихая: числа приходят, они правдоподобны, и неверны. Записи старых
версий остаются лежать — прошлый прогон должен воспроизводиться.

При смене `payload_type` (`_v2`) старые записи не читаются и не удаляются автоматически;
чистка — командой `scout cache drop`.
