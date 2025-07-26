# Qwen3 LoRA Finetuning

## Описание

Этот проект предназначен для дообучения модели Qwen3-1.7B с помощью LoRA на пользовательском датасете для задач reasoning и генерации.  
В ноутбуках реализованы этапы обучения, сохранения адаптеров, слияния с базовой моделью, а также пример инференса с возможностью выполнения кода.
Данная работа была выполнена в рамках студкемпа Яндекса и НГУ по обработке естественного языка.
Дообучение моделей проводилось на основе датасета сгенерированного более крупной моделью. Скрипт, реализующий эту идею можно найти по следующей ссылке: https://github.com/Siesher/Generator_for_reasoning.


## Структура

- `Qwen3_LoRA_on_gen_data.ipynb` — основной ноутбук для дообучения и инференса модели на основе пользовательского датасета.
- `Qwen3_LoRA_pretrain_on_gen_data.ipynb` — ноутбук для дообучения модели, которая ранее была обучена на датасете ZeroAgency/ru-thinking-reasoning-r1-v2, апосле дообучена по примеру с первым ноутбуком.
- `qwen3_adaptive_reasoning_dataset.jsonl` — датасет для обучения (формат JSONL).
- `merged_model/` — директория с итоговой слитой моделью (после обучения и слияния).
- `README.md` — описание проекта.

## Запуск

1. Установите зависимости:
    ```
    pip install transformers accelerate peft datasets sentencepiece tqdm bitsandbytes
    ```
2. (Для Google Colab) Смонтируйте Google Диск:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3. Укажите пути к датасету и директории для сохранения модели в ноутбуке.
4. Запустите ячейки для обучения.  
   После завершения обучения адаптеры LoRA будут сохранены в указанной директории.
5. Для инференса используйте пример из ноутбука — можно выполнять вопросы и получать ответы, включая выполнение кода.

## Пример инференса

```python
test_question = "Посчитай сумму всех чисел от 1 до 100."
run_inference_with_code_execution(
    inference_model,
    inference_tokenizer,
    test_question
)
```

## Требования

- Python 3.8+
- GPU (рекомендуется для обучения)
- Доступ к Google Colab или локальной машине с CUDA

## Контакты

Для вопросов и предложений пишите на loxterpoi@gmail.com.

## Полезные ссылки

Обученные модели можно найти на Hugging Face, а именно https://huggingface.co/Siesher

Обучение моделей построено на основе следующих статей:
- Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks
    URL: https://arxiv.org/abs/2211.12588
- Distilling mathematical reasoning capabilities into Small Language Models
    URL: https://arxiv.org/abs/2401.11864
- Towards Reasoning Ability of Small Language Models
    URL: https://arxiv.org/abs/2502.11569
- Skip-Thinking: Chunk-wise Chain-of-Thought Distillation Enable Smaller Language Models to Reason Better and Faster
    URL: https://arxiv.org/abs/2505.18642