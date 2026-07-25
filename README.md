# tik-tak

[![core](https://github.com/9ThLen/Tik-Tak/actions/workflows/core.yml/badge.svg)](https://github.com/9ThLen/Tik-Tak/actions/workflows/core.yml)
[![desktop](https://github.com/9ThLen/Tik-Tak/actions/workflows/desktop.yml/badge.svg)](https://github.com/9ThLen/Tik-Tak/actions/workflows/desktop.yml)
[![research](https://github.com/9ThLen/Tik-Tak/actions/workflows/research.yml/badge.svg)](https://github.com/9ThLen/Tik-Tak/actions/workflows/research.yml)

Метроном, який сам знаходить такт у музиці — з файлу або з мікрофона.

Мета — допомогти людям навчитися потрапляти в такт під час співу. Застосунок
визначає темп, знаходить долі й початок такту, і підказує їх візуально,
вібрацією та звуком.

## Стан

Активна розробка. Готові портативне C++17-ядро, офлайн-аналіз аудіофайлів,
декодування WAV/FLAC/MP3, кеш сітки долей, метроном і настільний стенд для
перевірки таймінгу. Python-реалізація залишається еталоном для досліджень і
автоматично порівнюється з C++.

📄 **[docs/PLAN.md](docs/PLAN.md)** — повний план реалізації: крос-платформенна
архітектура, алгоритм детекції такту, дорожня карта, ризики.

Ще не реалізовані онлайн-трекер для мікрофона, визначення сильної долі та
мобільні оболонки iOS/Android. Тобто репозиторій уже містить робоче ядро й
інструменти розробки, але ще не готовий користувацький застосунок.

## Швидкий старт

Потрібні Git, CMake 3.20+, компілятор C++17 і, бажано, Ninja. З кореня
репозиторію:

```sh
cmake -S core -B core/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build core/build
ctest --test-dir core/build --output-on-failure
```

Для Python-досліджень потрібен Python 3.11+:

```sh
python -m venv research/.venv
# Активуйте research/.venv відповідно до своєї оболонки.
python -m pip install -e "research[test]"
python -m pytest -q research
```

Перевірка паритету C++ і Python:

```sh
cmake -S tools/parity -B tools/parity/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build tools/parity/build
python tools/parity/check_parity.py
```

Настільний стенд:

```sh
cmake -S desktop -B desktop/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build desktop/build
desktop/build/tiktak render --bpm 137 --seconds 20 -o metronome.wav
```

На Windows виконуваний файл багатоконфігураційного генератора може лежати в
`desktop/build/RelWithDebInfo/tiktak.exe`. Докладніше — у
**[desktop/README.md](desktop/README.md)**.

## Режими

- **Файл** — імпортована мінусовка розбирається офлайн, метроном іде по точній
  сітці долей; синхронізація ідеальна за побудовою
- **Auto (мікрофон)** — метроном іде за живою музикою в реальному часі:
  репетиція гурту, живий інструмент
- **Manual + sync** — BPM задано вручну, але фаза вирівнюється під початок мелодії

Плюс сильна доля і розмір такту (4/4, 3/4, 6/8), субподіли долі, три незалежні
канали підказки. Усе на пристрої, офлайн.

## Стек

- **Ядро** — C++17 і CMake; портативний DSP, dr_libs для декодування
- **Настільний стенд** — C++17 і miniaudio
- **Дослідження** — Python, NumPy, mir_eval
- **Заплановано** — SwiftUI/AVAudioEngine/CoreHaptics для iOS; Compose/Oboe для
  Android; ONNX Runtime для визначення сильної долі
