# Noisy Channel Transmission

**Программная модель передачи данных по зашумлённому каналу связи (OFDM + QPSK + AWGN)**

---

## 📁 Проекты

| Проект | Описание | Перейти |
|--------|----------|---------|
| 📷 **Передача изображения** | Статичное изображение → OFDM → шум → восстановление | [`image_transmission/`](image_transmission/) |
| 🎬 **Передача видео** | MP4 видео → покадровая передача → графики → сравнение | [`video_transmission/`](video_transmission/) |

---

## ⚙️ Общие технологии

- **OFDM** (256 поднесущих) — мультиплексирование
- **QPSK модуляция** — 2 бита на символ
- **AWGN** — моделирование белого гауссовского шума
- **Квантование цветов** — простейшее сжатие (4 уровня серого)

---

## 🛠️ Запуск

### Передача изображения
```bash
cd image_transmission
pip install numpy matplotlib pillow
python image_transmission.py
```
### Передача видео
```bash
cd video_transmission
pip install numpy matplotlib pillow opencv-python
python video_transmission.py
```

---

## Результаты (для видео)

| Файл | Что показывает |
|------|----------------|
| ber_curve.png | График BER от SNR |
| 1_comparison.png | Сравнение исходного и восстановленного кадра |
| 2constellation.png | Созвездие QPSK |
| frame_comparison_1..5.png | Покадровое сравнение |

---

## Автор

**Виктория Комарова** (Vikitoria007)

## Лицензия

Проект распространяется под лицензией **MIT**.

Подробнее: [LICENSE](LICENSE)
