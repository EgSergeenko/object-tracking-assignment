from fastapi import FastAPI, WebSocket
from track_5 import track_data, country_balls_amount
import asyncio
import glob
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np



app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
print('Started')

def tracker_soft(el):
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
    return el
def xywh(list_a):
    x_l_up, y_l_up, x_r_down, y_r_down = list_a
    h = y_r_down - y_l_up
    w = x_r_down - x_l_up
    return [x_l_up, y_l_up, w, h]

def get_centroid(bbox: list):
    x_l_up, y_l_up, x_r_down, y_r_down = bbox
    x_center = x_l_up + int((x_r_down - x_l_up) / 2)
    y_center = y_l_up + int((y_r_down - y_l_up) / 2)
    return np.array([x_center, y_center])
def euclidian_metric(array_a, array_b):
    return np.sqrt(np.sum((array_a - array_b) ** 2))


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

    for el in track_data:
        await asyncio.sleep(0.5)
        # TODO: part 1
        el = tracker_soft(el)
        # TODO: part 2
        el = tracker_strong(el, tracker)
        # отправка информации по фрейму
        await websocket.send_json(el)
    print('Bye..')

