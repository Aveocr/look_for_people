# CLIP-ReID — Inference & Export (MSMT17)

Этот репозиторий содержит код для инференса и конвертации модели **CLIP-ReID** (ViT-B/16, обученной на MSMT17) в форматы ONNX и TensorRT.

> **Оригинальная статья и код:** [CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels](https://github.com/Syliz517/CLIP-ReID)

---

## Содержимое

```
clip-reid-msmt17/
├── config/                        # Конфигурационная система (yacs)
├── datasets/                      # Загрузчики датасета MSMT17
├── model/
│   ├── clip/                      # CLIP backbone (ViT-B/16)
│   └── make_model_clipreid.py     # Архитектура CLIP-ReID
├── processor/                     # Тренировочный процессор (stage 2)
├── utils/                         # Метрики, логгер, re-ranking
├── reid_inferece.ipynb            # Универсальный инференс (TRT / ONNX / CPU)
└── convert_to_onnx_trt.ipynb      # Конвертация PyTorch → ONNX → TensorRT
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install torch torchvision
pip install onnx onnxruntime-gpu
pip install pillow numpy yacs timm
# TensorRT (опционально, только для GPU-инференса):
pip install tensorrt pycuda
```

### 2. Получение весов

Скачайте обученные веса `MSMT17_clipreid_ViT-B-16_60.pth` из репозитория [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID) и положите их рядом с конфигом `clip-reid/configs/person/vit_clipreid.yml`.

### 3. Конвертация модели

Откройте и выполните [`convert_to_onnx_trt.ipynb`](convert_to_onnx_trt.ipynb). Ноутбук выполнит три шага:

| Шаг | Вход | Выход |
|---|---|---|
| 1. PyTorch → ONNX | `.pth` | `clipreid_msmt17.onnx` |
| 2. ONNX → TensorRT | `.onnx` | `clipreid_msmt17.trt` |
| 3. Верификация + бенчмарк | оба файла | таблица latency |

> **Важно:** для сборки TRT engine требуется NVIDIA GPU — модель использует `.cuda()` в `__init__`.

### 4. Инференс

Откройте [`reid_inferece.ipynb`](reid_inferece.ipynb). Бэкенд выбирается автоматически:

```
TensorRT  →  ONNX CUDA  →  ONNX CPU
```

---

## Параметры модели

| Параметр | Значение |
|---|---|
| Backbone | ViT-B/16 |
| Датасет | MSMT17 |
| Вход | `[B, 3, 256, 128]` float32 |
| Выход | `[B, 1280]` float32, L2-нормализован |
| camera\_num | 15 |
| num\_class | 1041 |
| mAP (статья) | 69.9% |
| Rank-1 (статья) | 86.7% |

### Нормализация входа (CLIP)

```python
mean = [0.48145466, 0.4578275,  0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]
```

---

## Производительность TensorRT (FP16)

Ориентировочные замеры на RTX 30xx/40xx:

| batch | TRT, ms | ONNX, ms | ускорение |
|---:|---:|---:|---:|
| 1 | ~1.5 | ~4.0 | ~2.7x |
| 8 | ~3.5 | ~12.0 | ~3.4x |
| 16 | ~6.0 | ~22.0 | ~3.7x |

> TRT engine привязан к конкретному GPU и версии TRT. При смене железа пересоберите engine заново (Шаг 2 в ноутбуке).

---

## Ссылки

- **CLIP-ReID (GitHub):** https://github.com/Syliz517/CLIP-ReID
- **Статья (AAAI 2023):** *CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels*, Siyuan Li et al.
- **CLIP (OpenAI):** https://github.com/openai/CLIP
