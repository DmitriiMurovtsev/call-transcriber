# Clef-Flash — спецификация

**Автор:** Cloudflare | **Лицензия:** Apache 2.0 | **Требование:** Ollama ≥ 0.35.1

## Обзор

Clef-Flash — мультимодальная **решенческая (decision) модель** на 9B параметров, дообученная на базе Qwen3.5-9B. Превращает состояние (`state`) и набор структурированных вопросов (`questions`) в вероятностные ответы.

**Ключевое отличие:** не-авторегрессионный однопассовый forward pass — все варианты всех вопросов оцениваются одновременно.

## Характеристики

| Параметр | Значение |
|---|---|
| Размер модели | 9B (11 GB) |
| Контекстное окно | 256K токенов |
| Модальности | текст + изображения (PNG, JPEG, WebP) |
| Точность BFCL | 98.8% |
| Медианная латентность | 38.8 ms |
| p95 латентность | 122.4 ms |

## Установка

```bash
ollama pull clef-flash
```

## API

Эндпоинт: `POST http://localhost:11434/v1/systemone`

### Запрос

| Поле | Описание |
|---|---|
| `model` | `"clef-flash"` |
| `state` | Текст/JSON для оценки (строка, объект или массив) |
| `images` | Опционально. Base64-кодированные изображения для всех вопросов |
| `questions` | 1–64 именованных вопросов |
| `keep_alive` | Опционально. Время удержания модели в памяти |

### Типы вопросов

| Тип | `criteria` | Возвращаемые поля |
|---|---|---|
| `choice` | Объект: `{"option": "description"}` (2–26 вариантов) | `choice`, `probabilities`, `confidence` |
| `noul` | Опционально: `{"true": "...", "false": "..."}` | `noul` (вероятность true) |
| `score` | Массив уровней от низшего к высшему (2–26) | `score`, `legend`, `probabilities`, `confidence` |

> `confidence` ∈ [0, 1] — степень концентрации вероятностей, **не** вероятность правильности.

### Ответ

```json
{
  "model": "clef-flash",
  "answers": {
    "urgent": { "type": "noul", "noul": 0.998 },
    "team": {
      "type": "choice",
      "choice": "technical",
      "probabilities": { "billing": 0.004, "technical": 0.992, "sales": 0.004 },
      "confidence": 0.947
    }
  },
  "usage": { "input_tokens": 486, "output_tokens": 3 }
}
```

## Пример: cURL

```bash
curl http://localhost:11434/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clef-flash",
    "state": "Checkout has been failing for every customer for the last hour.",
    "questions": {
      "urgent": { "type": "noul", "instructions": "Is this support request urgent?" },
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
          "billing": "Payments, invoices, and refunds",
          "technical": "Outages, errors, and configuration",
          "sales": "Plans and upgrades"
        }
      },
      "severity": {
        "type": "score",
        "instructions": "How severe is the customer impact?",
        "criteria": ["No impact", "Minor", "Major", "Critical"]
      }
    }
  }'
```

## Пример: Python (TypeSafe SDK)

```bash
pip install typesafe-sdk
export TYPESAFE_BASE_URL=http://localhost:11434
export TYPESAFE_API_KEY=ollama
export TYPESAFE_DEFAULT_MODEL=clef-flash
```

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

questions = {
    "urgent": Noul(instructions="Is this support request urgent?"),
    "team": Choice(
        instructions="Which team should handle this request?",
        criteria={
            "billing": "Payments, invoices, and refunds",
            "technical": "Outages, errors, and configuration",
        },
    ),
    "severity": Score(
        instructions="How severe is the customer impact?",
        criteria=["No impact", "Minor", "Major", "Critical"],
    ),
}

with TypeSafeClient(timeout=120) as client:
    result = client.system_one(state="Checkout has been failing...", questions=questions)

print(result.nouls["urgent"].noul)     # 0.998
print(result.choices["team"].choice)   # technical
print(result.scores["severity"].score) # 2.85
```

## Сценарии использования

| Задача | Вы определяете | Получаете |
|---|---|---|
| Маршрутизация запроса | Назначения и условия | Выбранное назначение + вероятности |
| Проверка условия | Да/нет вопрос + доказательства | True/False + вероятности |
| Применение политики | Правила и допустимые исходы | Типизированное решение |
| Гейтирование шага агента | Условие + proposed tool call | Go/No-go + вероятность |

## Бенчмарки (выборочные)

| Benchmark | Clef-flash | Clef (27B) | Jev |
|---|---|---|---|
| BFCL | 98.8 | 98.5 | 95.8 |
| API-Bank | 93.1 | 91.9 | 88.2 |
| Home appliance sim | 97.7 | 83.0 | 52.3 |
| Median latency | 38.8 ms | 209.3 ms | 524.1 ms |

Clef-Flash в ~5 раз быстрее Clef (27B) при сопоставимой точности.

## Ссылки

- Ollama: https://ollama.com/library/clef-flash
- Документация: https://docs.ollama.com/capabilities/decision
- Python SDK: https://github.com/ollama/ollama-python
