from fastapi import FastAPI, WebSocket
from track_3 import track_data, country_balls_amount
import asyncio
import glob
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from PIL import Image

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
    w = y_r_down - y_l_up
    h = x_r_down - x_l_up
    return [x_l_up, y_l_up, w, h]

def tracker_strong(el, tracker):
    bbs = []
    frame = cv2.imread('frame/' + str(el['frame_id']) + '.png')
    cv2.imshow('image', frame)
    for i in range(len(el['data'])):
        if any(el['data'][i]['bounding_box']):
            bbs.append((xywh(el['data'][i]['bounding_box']), 1, 0))

    tracks = tracker.update_tracks(bbs, frame=frame)  # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )

    # for track in tracks:
    #     for i in range(len(el['data'])):
    #     # print(xywh(el['data'][i]['bounding_box']))
    #     # print(track.to_ltrb())
    #         if not track.is_confirmed():
    #                 continue
    #         if not any(el['data'][i]['bounding_box']):
    #             el['data'][i]['track_id'] = None
    #             continue
    #         if xywh(el['data'][i]['bounding_box'])[0] == track.to_ltrb()[0] and \
    #                 xywh(el['data'][i]['bounding_box'])[1] == track.to_ltrb()[1]:
    #             el['data'][i]['track_id'] = track.track_id
    #     # if not track.is_confirmed():
    #     #         continue
    #     # track_id = track.track_id
    #     # el['data'][i]['track_id'] = track.track_id
    #     # ltrb = track.to_ltrb()
    for i in range(len(el['data'])):
        pass
    #TODO для всех эл-тов, у которых есть бб, присв ID
    # когда у эл нет бб, заносим в список
    # среди треков дипсорт поискать свободные ИД и рандомно раскидать по списку с пустыми бб


    for i, track in enumerate(tracks):
            if not track.is_confirmed():
                    continue
            if not any(el['data'][i]['bounding_box']):
                el['data'][i]['track_id'] = None
                continue
            el['data'][i]['track_id'] = track.track_id

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
