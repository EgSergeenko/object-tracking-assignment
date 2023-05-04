from fastapi import FastAPI, WebSocket
from track_3 import track_data, country_balls_amount
import asyncio
import glob
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np


from metrics import MetricsAccumulator

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


def xywh(list_a):
    x_l_up, y_l_up, x_r_down, y_r_down = list_a
    h = y_r_down - y_l_up
    w = x_r_down - x_l_up
    return [x_l_up, y_l_up, w, h]


def tracker_strong(el, tracker):
    bbs = []
    im = cv2.imread('frame/' + str(el['frame_id']) + '.png')
    frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for i in range(len(el['data'])):
        if any(el['data'][i]['bounding_box']):
            bbs.append((xywh(el['data'][i]['bounding_box']), 1, 0))

    tracks = tracker.update_tracks(bbs, frame=frame)  # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )

    bb_track = {i.track_id: get_centroid(i.to_ltrb()) for i in tracks}

    for i in range(len(el['data'])):
        if el['data'][i]['bounding_box']:
            if len(bb_track):
                centr_el = get_centroid(el['data'][i]['bounding_box'])
                id, _ = min(bb_track.items(),
                    key=lambda x: euclidian_metric(x[1], centr_el))

                el['data'][i]['track_id'] = id
                del bb_track[id]

    return el


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    tracker = DeepSort(max_age=5)
    await websocket.send_text(str(country_balls))

    accumulator = MetricsAccumulator()

    for el in track_data:
        await asyncio.sleep(0.5)
        # TODO: part 1
        if el['frame_id'] == 1:
            id_info = {}
            num = 0
        el_soft, id_info, num = tracker_soft(el, id_info, num)

        # TODO: part 2
        el_strong = tracker_strong(el, tracker)
        # отправка информации по фрейму
        await websocket.send_json(el_strong)
        accumulator.update(el_strong)
    print('Bye..')

