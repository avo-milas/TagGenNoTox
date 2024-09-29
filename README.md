<a id="readme-top"></a>
  <h1 align="center">TagGenNoTox</h1>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contacts">Contacts</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Мы представляем инновационную модель для автоматической классификации видеоконтента по тегам, использующую данные разных типов, что существенно упрощает поиск и анализ контента.

С помощью YAMNET мы эффективно извлекаем аудио, а OpenCV позволяет выделять ключевые кадры, что дает возможность анализировать визуальные аспекты видео. Мы также интегрируем название и описание видео, создавая многомодальные эмбеддинги из всех источников данных. Эти эмбеддинги служат основой для обучения нашей нейронной сети, обеспечивая высокую точность классификации тегов как первого, так и второго уровня.

Уникальность нашего подхода заключается в синергии метаданных, что позволяет улучшить качество предсказаний и адаптироваться к разнообразным видам контента, обеспечивая более глубокое понимание его содержания.

Стек решения: Python, YAMNET, OpenCV, SentenceTransformers, PyTorch.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

**Для выполнения инференса**:

1. В ноутбуке `inference.ipynb` указать путь до датасета с тестовыми данными, который будет лежать в папке `data`:
```python
test_data = pd.read_csv("...путь...")
```

2. Запустить все ячейки в ноутбуке
3. Результаты будут лежать в файле `submission_test_data.csv`

- **model:**
`cnn_model.py` - CNN модель для двух уровней тегирования

Параметры модели, полученные после обучения:
`mlb_first_level.pkl`
`mlb_second_level.pkl`
`model_first_level.pth`
`model_second_level.pth`
`tag_mapping.pkl`

- **utils:**
  
`data_preprocessing.py` - лемматизация, удаление стоп-слов, токенизация и создание эмбеддингов

`tag_extraction.py` - извлечение тегов на разных уровнях, фильтрация тегов 2 уровня по 1

- **notebooks:**
  
`baseline.ipynb` - обучение и тестирование модели

`EDA.ipynb` - data exploration

`inference.ipynb` - запуск модели на тестовых данных

`video_extracting` - извлечение видео из датасетов

- **data:**
  
`train_data_categories.csv` - обучающая выборка

`IAB_tags.csv` - таксономия тегов

`data_proc.pkl` - обработанные данные

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contacts

Alina Salimova - [@avo_milas](https://t.me/avo_milas) - avo_milas@mail.ru


Project Link: [https://github.com/avo-milas/TagGenNoTox](https://github.com/avo-milas/TagGenNoTox)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
