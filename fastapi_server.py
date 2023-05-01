from fastapi import FastAPI, WebSocket
from track_5 import track_data, country_balls_amount
import asyncio
import glob
import numpy as np

app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
print('Started')


def get_centroid(bbox: list):    
    x_l_up, y_l_up, x_r_down, y_r_down = bbox
    x_center = x_l_up + int((x_r_down - x_l_up) / 2)
    y_center = y_l_up + int((y_r_down + y_l_up) / 2)
    return  np.array([x_center, y_center])


def euclidian_metric(array_a, array_b):
    return np.sqrt(np.sum((array_a - array_b) ** 2))


def tracker_soft(el, id_info, num):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """
    tracks = {}
    # присваиваем айдишники от 0 до кол-ва кантриболов на первом кадре
    if el['frame_id'] == 1:
        for num in range(len(el['data'])):
            el['data'][num]['track_id'] = num
            if any(el['data'][num]['bounding_box']):
                tracks[num] = get_centroid(el['data'][num]['bounding_box'])
        num += 1

        return el, tracks, num

    for i in range(len(el['data'])):
        if any(el['data'][i]['bounding_box']):            
            center_coordinates = get_centroid(el['data'][i]['bounding_box'])
            if len(id_info) != 0:
                id, _ = min(id_info.items(),
                            key=lambda x: euclidian_metric(x[1], center_coordinates))
                del id_info[id]
            else:
                id = num
                num += 1
            el['data'][i]['track_id'] = id
            tracks[id] = center_coordinates
        else:
            el['data'][i]['track_id'] = num
            num += 1

    return el, tracks, num


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true для первого прогона, на повторном прогоне можете читать фреймы из папки
    и по координатам вырезать необходимые регионы.
    TODO: Ужасный костыль, на следующий поток поправить
    """
    return el


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))
    for el in track_data:
        await asyncio.sleep(0.5)
        # TODO: part 1
        if el['frame_id'] == 1:
            id_info = {}
            num = 0
        el, id_info, num = tracker_soft(el, id_info, num)
        # TODO: part 2
        # el = tracker_strong(el)
        # отправка информации по фрейму
        await websocket.send_json(el)
    print('Bye..')
