import amrlib
import os
import wget
import tarfile
import requests
import networkx as nx
import matplotlib.pyplot as plt
import time
from amrlib.evaluate.alignment_scorer import AlignmentScorer
from amrlib.evaluate.bleu_scorer import BLEUScorer


def format_time(seconds):
    """
    Форматирование секунд к виду ЧЧ:ММ:СС

      Возвращает строку отформатированного времени
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


def url_exists(url):
    '''
    Функция проверяет существование web-ссылки

    Если ссылка найдена - возвращает True, иначе False
    '''
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"Не удалось найти указанный адрес: {url} По причине:: {e}")
        return False


def file_exists(filepath):
    '''
    Функция проверяет существование файла

    Если файл найден - возвращает True, иначе False
    '''
    return os.path.isfile(filepath)


def models_installed(model_dir):
    '''
    Функция проверяет установлены или нет требуемые модели

    Если модели не установлены - пытается их установить.
    Если установить удалось - возвращает True, иначе False
    '''

    # Проверка наличия моделей в проекте
    if (file_exists(model_dir + '/model_parse_xfm_bart_base-v0_1_0/pytorch_model.bin')
            and file_exists(model_dir + '/model_parse_xfm_bart_base-v0_1_0/pytorch_model.bin')):
        return True

    # Если моделей нет - будем их загружать из github

    # Проверка ссылки на модель arse_xfm_bart_base
    url_parse_xfm_bart_base = ('https://github.com/bjascob/amrlib-models/releases/download/'
                               'parse_xfm_bart_base-v0_1_0/model_parse_xfm_bart_base-v0_1_0.tar.gz')
    url_parse_xfm_bart_base_exists = False
    if url_exists(url_parse_xfm_bart_base):
        url_parse_xfm_bart_base_exists = True
    else:
        url_parse_xfm_bart_base_exists = False

    # Проверка ссылки на модель generate_t5wtense_bart_large
    url_generate_t5wtense_bart_large = ('https://github.com/bjascob/amrlib-models/releases/download/'
                                        'model_generate_t5wtense-v0_1_0/model_generate_t5wtense-v0_1_0.tar.gz')
    url_generate_t5wtense_bart_large_exists = False
    if url_exists(url_generate_t5wtense_bart_large):
        url_generate_t5wtense_bart_large_exists = True
    else:
        url_generate_t5wtense_bart_large_exists = False

    # Проверка наличия директории для загружаемых моделей. Если нет - создаем
    model_dir_exists = True
    if not os.path.isdir(model_dir):
        try:
            os.makedirs(model_dir, exist_ok=True)
        except Exception as e:
            model_dir_exists = False
            print(f'Не удалось создать директорию: {model_dir_exists} по причине: {e}')

    # Загрузка моделей из сети
    files_downloaded = True
    if url_parse_xfm_bart_base_exists and url_generate_t5wtense_bart_large_exists and model_dir_exists:
        try:
            wget.download(url=url_parse_xfm_bart_base, out=model_dir)
        except Exception as e:
            files_downloaded = False
            print(f'Не удалось скачать файл: {url_parse_xfm_bart_base} по причине: {e}')
        try:
            wget.download(url=url_generate_t5wtense_bart_large, out=model_dir)
        except Exception as e:
            files_downloaded = False
            print(f'Не удалось скачать файл: {url_generate_t5wtense_bart_large} по причине: {e}')
    else:
        files_downloaded = False

    # Распаковка моделей в указанную директорию
    models_installed = True
    if files_downloaded:
        try:
            target_file = model_dir + '/model_parse_xfm_bart_base-v0_1_0.tar.gz'
            with tarfile.open(target_file, 'r:gz') as tar:
                tar.extractall(path=model_dir)
        except Exception as e:
            models_installed = False
            print(f'Не удалось распаковать файл: {target_file} по причине: {e}')

        try:
            target_file = model_dir + '/model_generate_t5wtense-v0_1_0.tar.gz'
            with tarfile.open(target_file, 'r:gz') as tar:
                tar.extractall(path=model_dir)
        except Exception as e:
            models_installed = False
            print(f'Не удалось распаковать файл: {target_file} по причине: {e}')
    else:
        models_installed = False

    return models_installed


def visualize_amr_graph(amr_str):
    graph = nx.DiGraph()
    lines = amr_str.strip().split('\n')
    for line in lines:
        if not line.startswith('#'):
            parts = line.split()
            if len(parts) > 2:
                parent = parts[0].strip('()')
                relation = parts[1]
                child = parts[2].strip('()')
                graph.add_edge(parent, child, label=relation)
    pos = nx.spring_layout(graph)
    edge_labels = {(u, v): d['label'] for u, v, d in graph.edges(data=True)}
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color='lightblue',
            font_size=8, font_weight='bold', arrowsize=9)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    plt.show()


# Посчитаем сколько времени уйдет на всю работу
start_time_all = time.time()


print()
print('Проверка наличия натренированных моделей. Если их нет произойдет загрузка.')
start_time = time.time()
model_dir = '../../models/amrlib_models'
if not models_installed(model_dir):
    print('Модели не установлены!')
    exit()
print("- Время проверки наличия/распаковки моделей: ", format_time(time.time() - start_time))

print()
print('Parse Models: Загрузка модели для синтаксического анализа - parse_xfm_bart_base')
start_time = time.time()
stog = amrlib.load_stog_model(model_dir=f'{model_dir}/model_parse_xfm_bart_base-v0_1_0', batch_size=12, num_beams=4)
print()
print('Популярные цитаты из Шекспира:')
sentence = ('Better Three Hours Too Soon Than A Minute Too Late.'
            'My Words Fly Up, My Thoughts Remain Below. Words Without Thoughts Never To Heaven Go.'
            'Brevity Is The Soul Of Wit.'
            'All That Glitters Is Not Gold.'
            'No Legacy Is So Rich As Honesty.'
            )
sentences = sentence.split('.')
sentences = [s.strip() for s in sentences if s.strip()]
for s in sentences:
    print(s)
print("- Время объявления модели : parse_xfm_bart_base: ", format_time(time.time() - start_time))

print()
print('Создание графов')
start_time = time.time()
graphs = stog.parse_sents([sentence])
for graph in graphs:
    print(graph)
print("- Время создания графов: ", format_time(time.time() - start_time))

print()
print('Вывод и визуализация графов')
start_time = time.time()
if not graphs:
    print('Графы не сгенерированы.')
else:
    print('Parse Model:')
    amr_graph_str = graphs[0]
    visualize_amr_graph(amr_graph_str)
print("- Время вывода и создания картинки графов: ", format_time(time.time() - start_time))


print()
print('Generate Models: Объявление модели для генерации текста - generate_t5wtense и генерация предложений')
gtos = amrlib.load_gtos_model(model_dir=f'{model_dir}/model_generate_t5wtense-v0_1_0')
print()
# Генерация предложения
start_time = time.time()
generated_sents, _ = gtos.generate(graphs)
print('Сгенерированный текст:')
gener_sents = generated_sents[0].split('.')
gener_sents = [s.strip() for s in gener_sents if s.strip()]
for s in gener_sents:
    print(s)
print("- Время генерации предложений: ", format_time(time.time() - start_time))

print()
print('Метрики оценки:')
bleu_scorer = BLEUScorer()
bleu_score, ref_len, hyp_len = bleu_scorer.compute_bleu(list([sentence]), list(generated_sents))
bleu_score_percent = bleu_score*100
print(f'BLEU score: {bleu_score_percent:.2f}%')

scorer = AlignmentScorer(list([sentence]), list(generated_sents))
recall_scores = scorer.recall_scores                        # можно и отдельно
precision_scores = scorer.precision_scores                  # можно и отдельно
get_precision_recall_f1 = scorer.get_precision_recall_f1()  # можно и отдельно
print(f'Alignment scores: {scorer}')                        # все вместе
print()
print("- Время всей работы: ", format_time(time.time() - start_time_all))
