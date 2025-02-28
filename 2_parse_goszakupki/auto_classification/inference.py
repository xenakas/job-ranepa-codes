import numpy as np
import torch
from bpemb import BPEmb
from utils_d.ml_utils import pad_text
from pytorch_seq2seq import CNN

use_cuda = True
y_dim = 1000
output_len = 25
bpemb_ru = BPEmb(lang="ru", dim=50, vs=y_dim)
cut_zero = False


def load_model(model_name):
    emb_matrx = np.array([bpemb_ru[w] for w in bpemb_ru.words])
    model = CNN(pretrained_embeddings=emb_matrx, output_chars=25)
    model.load_state_dict(torch.load(f"nn_models/{model_name}"))
    model.eval()
    return model


def preprocess_text(text, max_len):
    text = bpemb_ru.encode_ids(text)
    text = pad_text(text, padding_text=0, max_len=max_len)[:max_len]
    return text


def inference(model, text, additional_text):
    text = preprocess_text(text, max_len=1000)
    additional_text = preprocess_text(additional_text, max_len=6)
    model_input = text + additional_text
    model_input = np.array([model_input])
    model_input = torch.tensor(model_input)
    if use_cuda:
        model_input = model_input.cuda()
    with torch.set_grad_enabled(False):
        preds, _ = model(model_input)
    preds = preds[0]
    # preds = torch.sigmoid(preds).data > 0.5
    preds = preds.cpu()
    preds = preds.numpy()
    preds = preds.reshape(25, 1000)
    if cut_zero:
        preds = np.argmax(preds[:, 1:], axis=1)
        output = bpemb_ru.decode_ids(preds + 1)
    else:
        preds = np.argmax(preds, axis=1)
        output = bpemb_ru.decode_ids(preds)
    print(output)
    return output


if __name__ == "__main__":
    text = """

        Владельцев по ПТС: 1
        Пробег: 81000 км
        Состояние: не битый
        Руль: левый
        Привод: передний
        Цвет: белый
        Объём двигателя: 2.5
        Модель: Teana
        Марка: Nissan
        Год выпуска: 2012
        Тип кузова: седан
        Тип двигателя: бензин
        Коробка передач: вариатор
        Мощность двигателя: 185 л.с.
        VIN или номер кузова: Z8NBBUJ3*CS****70

    Адрес: Варшавское ш., д. 170Г, стр. 3
    Показано из
    * Автомобиль в очень хорошем состоянии.
    * Лакокрасочное покрытие без повреждений.
    * Коробка и двигатель работают идеально.
    * Технически автомобиль полностью исправен.
    * Салон чистый, не прокуренный. Вложений не требует.
    * Второй комплект ключей и комплект резины в наличии.

    Все автомобили, представленные в продаже проходят диагностику по
    30 параметрам.
    Также мы предоставляем гарантию юридической чистоты в письменной форме.

    При обмене вашего старого автомобиля вы можете получить скидку в размере
    до 50000 рублей на данный автомобиль.

    Помимо покупки за наличные или обмена по Trade-In, вы можете
    воспользоваться
    одной из нескольких выгодных программ автокредитования от наших
    банков-партнеров.
    - Оформление кредита по двум документам, за ЧАС
    - Первоначальный взнос от 0%
    - Процентная ставка от 9 до 17% годовых
    - КАСКО не обязательно

    Приглашаем на бесплатный тест-драйв!
    Более подробная информация на нашем сайте.

        Салон
            Кожа
        Обогрев
            Передних сидений
            Зеркал
        Электростеклоподъемники
            Передние и задние
        Электропривод
            Зеркал
        Помощь при вождении
            Датчик дождя
            Датчик света
            Парктроник задний
            Камера заднего вида
            Круиз-контроль
            Бортовой компьютер
        Противоугонная система
            Сигнализация
        Активная безопасность
            Антиблокировочная система тормозов (ABS)
            Антипробуксовочная система (ASR)
            Система курсовой устойчивости (ESP/ESC/DSC)
        Мультимедиа и навигация
            CD/DVD/Blu-ray
            GPS-навигатор
        Фары
            Ксеноновые
            Противотуманные
            Омыватели фар
        Шины и диски
            17"

    """
    additional_text = "Год выпуска"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model("2019_3_12_default")
    model = model.to(device)
    inference(model, text, additional_text)
