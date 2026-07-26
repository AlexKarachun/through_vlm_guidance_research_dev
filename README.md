# VLM-guided image generation


<img src="figures/guided_generation_pipeline.png?v=wave5-cfg11" alt="VLM-guided generation pipeline" width="100%">

## О проекте

Мы пытаемся повысить корректность генерируемых диффузионными моделями изображений с помощью нашего VLM guided пайплайна семплинга. В целом наш подход похож на classifier guidance, но вместо классификатора мы используем VLM. Мы генерируем изображение, вычисляем VLM loss и проводим градиент обратно до выбранного латента сквозь диффузионную генерацию.После обновления `X_t` изображение генерируется повторно.

## Установка

```bash
git clone https://github.com/AlexKarachun/through_vlm_guidance_research.git
cd through_vlm_guidance_research
conda env create -f environment.yml
conda activate through_guidance
```

## Запуск

### Vanilla Stable Diffusion 1.5

Single prompt:

```bash
HYDRA_FULL_ERROR=1 python run.py \
  pipeline=sd15 \
  generation=single_prompt \
  'generation.prompt=A bear does a handstand in the park' \
  pipeline.params.n_inference_steps=50
```

Multiple prompts:

```bash
HYDRA_FULL_ERROR=1 python run.py \
  pipeline=sd15 \
  generation=multi_prompt \
  generation.prompts_file=datasets/whoops50.txt \
  pipeline.params.n_inference_steps=50
```

### Guided Stable Diffusion 1.5

Single prompt:

```bash
HYDRA_FULL_ERROR=1 python run.py \
  pipeline=guided_sd15 \
  generation=single_prompt \
  'generation.prompt=A bear does a handstand in the park' \
  pipeline.params.n_inference_steps=50 \
  pipeline.params.cfg_scale=15 \
  'pipeline.guidance.steps_to_guide=[10,10,10,10]' \
  pipeline.optimizer.lr=4e-5
```

Multiple prompts:

```bash
HYDRA_FULL_ERROR=1 python run.py \
  pipeline=guided_sd15 \
  generation=multi_prompt \
  generation.prompts_file=datasets/whoops50.txt \
  pipeline.params.n_inference_steps=50 \
  pipeline.params.cfg_scale=15 \
  'pipeline.guidance.steps_to_guide=[10,10,10,10]' \
  pipeline.optimizer.lr=4e-5
```


### Judge
Для подсчета метрик alignment - верность изображния и quality - визуальная корректность изображения мы используем judge пайплайн
```bash
HYDRA_FULL_ERROR=1 python run.py \
  pipeline=qwen3_judge \
  generation=multi_prompt \
  pipeline.model.model_id=Qwen/Qwen3-VL-8B-Instruct \
  pipeline.params.judgment_results_file=generations/judgment_qwen3vl8b.csv
```

## Потребление VRAM

| Pipeline | Модель | VRAM |
|---|---|---:|
| SD1.5 | Stable Diffusion 1.5 | ≈11 GB |
| Guided SD1.5 | SD1.5 + Qwen3-VL-2B | ≈23 GB |
| Judge | Qwen3-VL-8B | ≈18 GB |
| Judge | Qwen3-VL-32B | ≈64 GB |


<br>

Эксперименты выполнялись на cuda 13+ на
- RTX 3090 24 GB vram
- RTX 4090 24 GB vram
- RTX 5090 32 GB vram
- RTX PRO 6000 96 GB vram


<br>
Технические отчеты о ходе работы можно найти здесь

- <a href="https://alexkarachun.github.io/through_vlm_guidance_research/experiments/wave_5/external/wave_5_report.html">Wave 5 report</a>




<!-- 
done 
- сделать график запусков в осях alignment/quality
- построить p_yes траектории для лучших запусков с разных таймстепов. строить средние с дисперсией, чтобы было понятно, есть ли разница с какого шага начинать


todo
- увеличим число итераций
- уменьшим learning rate
- проследим за градиентами (возможно взрываются)



- оформить гитхаб (through_vlm_guidance_research, RAEDME - описание, картинки, ссылку на отчет)








потребление vram:
- sd1.5 ~11gb
- guided_sd1.5 ~23gb
- judge Qwen/Qwen3-VL-8B-Instruct: ~18gb
- judge Qwen/Qwen3-VL-32B-Instruct: ~64gb

Автор исполнял код на 
- RTX 3090 24gb vram
- RTX 4090 24gb vram
- RTX 5090 32gb vram
- RTX PRO 6000 96gb vram




git clone https://github.com/AlexKarachun/through_vlm_guidance_research_dev.git
cd through_vlm_guidance_research_dev
conda env create -f environment.yml
conda activate through_guidance


-->
