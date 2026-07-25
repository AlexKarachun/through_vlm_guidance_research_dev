# through_guidance

setup
```shell
git clone https://github.com/AlexKarachun/through_vlm_guidance_research_dev.git
cd through_vlm_guidance_research_dev
conda env create -f environment.yml
conda activate through_guidance
```


```shell
HYDRA_FULL_ERROR=1 python run.py\
    pipeline=sd15\
    generation=single_prompt\
    pipeline.params.n_inference_steps=10

HYDRA_FULL_ERROR=1 python run.py\
    pipeline=sd15\
    generation=multi_prompt\
    pipeline.params.n_inference_steps=10

HYDRA_FULL_ERROR=1 python run.py\
    pipeline=guided_sd15\
    generation=single_prompt\
    pipeline.params.n_inference_steps=10\
    'pipeline.guidance.steps_to_guide=[3]'

HYDRA_FULL_ERROR=1 python run.py\
    pipeline=guided_sd15\
    generation=multi_prompt\
    pipeline.params.n_inference_steps=10\
    'pipeline.guidance.steps_to_guide=[2]'\
    pipeline.optimizer.lr=1e-4





HYDRA_FULL_ERROR=1 python run.py\
    pipeline=guided_sd15\
    generation=single_prompt\
    'generation.prompt=A bear does a handstand in the park'\
    pipeline.params.n_inference_steps=50\
    'pipeline.guidance.steps_to_guide=[20]'\
    pipeline.optimizer.lr=1e-1\
```









<!-- 
done 
- сделать график запусков в осях alignment/quality
- построить p_yes траектории для лучших запусков с разных таймстепов. строить средние с дисперсией, чтобы было понятно, есть ли разница с какого шага начинать


todo
- использовать более сильного judge
- запустить эксп на слишком много итераций гайденса - возмжоно для маленьких lr vlm_loss не будет так асцилировать
- возможно у нас взрываются градиенты. нарисовать график плотности чисел градиентов, нарисовать график норм градиентов при backward 50->0.



- написать отчет для Айбека
- оформить гитхаб (through_vlm_guidance_research, RAEDME - описание, картинки, ссылку на отчет)

что я хочу узнать от графиков
- улучшают ли шаги гайденса финальный результат для лучшего сочетания парамтеров - p_yes / guidance_iteration ; выставить intermidiate finals в ряд
- насколько наш лучший пайплайн улучшает картинки - сравнение прироста alignment (pisitive/negative bars in barplot) и топ лучших и худших пар картинок (со ссылками на файлы картинок)









how guided pipeline works
- vlm guidance via vlm loss
- ddpm noise caching для того, чтобы сравнивать корректировку траектории, а не "разные сиды"


потребление vram:
- sd1.5 ~11gb
- guided_sd1.5 ~23gb
- judge Qwen/Qwen3-VL-8B-Instruct: ~18gb
- judge Qwen/Qwen3-VL-32B-Instruct: ~18gb

Автор исполнял код на 
- RTX 3090 24gb vram
- RTX 4090 24gb vram
- RTX 5090 32gb vram
- RTX PRO 6000 96gb vram


-->




