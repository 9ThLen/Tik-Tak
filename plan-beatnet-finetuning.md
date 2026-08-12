# Альтернативний план: адаптація BeatNet без створення нового audio frontend

> Статус: окремий альтернативний research plan, підготовлений за станом репозиторію на `223f894`.
>
> [Основний план створення/вибору моделі](plan.md) не змінюється. Цей документ не скасовує його й не дає дозволу запускати дороге навчання, змінювати production-код або відкривати locked split без окремого рішення.

## 1. Рішення в одному абзаці

Основний шлях цієї альтернативи — не проєктувати новий audio frontend, а адаптувати вже портовану причинну BeatNet до цільового домену Tik-tak. Незмінена архітектура BeatNet уже дає 50-fps класи `beat/downbeat/non-beat`, а локальний код перетворює їх на узгоджені `P(any beat)=P(beat)+P(downbeat)` і `P(downbeat)`, після чого наявні `LiveTracker` та `BarTracker` відновлюють tactus grid, bar onset, bar position і tactus grouping. Навчання ескалюється від калібрування й head-only до часткового та повного fine-tune; реальні paired clean↔microphone записи є головним доменним матеріалом. Новий neural head для subdivision/`meter_family` дозволяється лише після M0c і нового позитивного метричного гейта, якщо BeatNet уже дає добрі tactus/downbeat, але наявний metrical decoder інформаційно недостатній. Denominator не вважається акустичним передбаченням. Нова audio-архітектура в цей план не входить.

## 2. Чому цей шлях справді коротший

| Уже існує | Що це прибирає з роботи |
|---|---|
| Закріплені BeatNet checkpoints і provenance у [models/manifest.json](models/manifest.json) | Не потрібно шукати початкові ваги або вигадувати модельний формат |
| Точний exporter [models/export_beatnet.py](models/export_beatnet.py) | Донавчені ваги з незмінними tensor names/shapes можна конвертувати в наявний `TTBN v1` |
| Причинний C++ forward-pass у [core/src/ml/beatnet.hpp](core/src/ml/beatnet.hpp) | Не потрібні ONNX Runtime, новий inference wrapper або нові operators для базової гілки |
| `BeatNetActivation` уже формує `beat_total = beat_class + downbeat_class` | Downbeat є підмножиною загальної beat evidence за контрактом inference |
| `LiveTracker::observe(time, beat, downbeat)` у [core/src/tracking/live.hpp](core/src/tracking/live.hpp) | Не потрібно будувати новий причинний beat decoder |
| `BarTracker` і `resolveMeter` у [core/src/tracking/bar.hpp](core/src/tracking/bar.hpp) та [core/src/analysis/downbeat.hpp](core/src/analysis/downbeat.hpp) | Bar phase і перший grouping/meter baseline уже є |
| Annotation/eval код у [research/eval/annotations.py](research/eval/annotations.py) і [research/eval/downbeat.py](research/eval/downbeat.py) | Не створюється паралельний формат або другий harness |
| Upstream BeatNet 1.2.0 уже містить training pipeline | Локально потрібен тонкий adapter та project-specific evaluation, а не ще один trainer з нуля |

Найбільша перевага зникає, якщо одразу змінити input features, кількість LSTM layers або output heads. Тому незмінена BeatNet є default, а кожна несумісна зміна повинна спершу довести, що не може бути замінена fine-tuning або post-processing.

## 3. Межі альтернативи

### Входить

- fine-tuning оригінальної BeatNet CRNN;
- calibration і checkpoint selection за продуктовими beat/bar/meter метриками;
- реальні paired clean↔microphone дані та перевірені аугментації;
- reuse наявних `LiveTracker`, `BarTracker`, annotation і evaluation contracts;
- за потреби маленький metrical/subdivision head поверх представлення BeatNet або accepted tactus sequence;
- export/parity для донавчених ваг, on-device перевірка і locked confirmation;
- frozen BeatNet+ як research comparator, якщо його артефакти та умови використання зафіксовані.

### Не входить

- harmonic-aware CNN, TCN, GRU/attention matrix або інший новий audio encoder;
- raw-waveform frontend;
- окрема BPM head;
- порт upstream particle filter до того, як oracle покаже decoder bottleneck;
- автоматичне редагування чи скорочення [plan.md](plan.md);
- shipping BeatNet/BeatNet+ до окремого рішення щодо ліцензії ваг і App Store/DRM;
- tuning на locked split.

Якщо ця альтернатива вичерпана й не проходить preregistered gate, результатом є відтворюваний failure report. Перехід до власної архітектури відбувається лише окремим рішенням за основним планом.

## 4. Продуктовий output contract

Терміни мають ті самі значення, що й в основному плані:

```text
tactus timestamps + confidence
downbeat/bar-onset timestamps + confidence
tactus position for each accepted tactus
tactus-beats-per-bar/grouping per stable segment + confidence
subdivisions-per-tactus: simple | compound | unknown
meter_family or unknown
canonical_time_signature or unknown + notation_basis
meter-change event / acquisition/unknown state
BPM derived from accepted tactus intervals
```

- У Metrical v1 `strong beat` означає лише downbeat/bar onset.
- Вторинний акцент, наприклад третя доля у 4/4, виводиться з `bar_position + meter`; він не є окремим BeatNet class.
- `tactus_beats_per_bar=2` разом із трійковим subdivision дає compound duple, але не визначає, чи нотація є 6/8, 6/4 або 6/16.
- `meter_family` — обов'язковий акустичний вихід. `canonical_time_signature` — вторинний і завжди має `notation_basis = annotated | user_supplied | corpus_convention | inferred`; denominator із конвенції чи від користувача не зараховується моделі.
- Система може відповідати `unknown`; вона не може підміняти meter константою 4.
- Правильний BPM або високий beat F не компенсує помилкові downbeat, bar phase чи meter.

## 5. Цільова архітектура

```text
microphone audio
      │
      ▼
existing BeatNet features: 272 values, 50 fps
      │
      ▼
fine-tuned existing BeatNet CRNN: Conv1d + 2×LSTM(150) + 3-way softmax
      │
      ├── beat_total = P(beat class) + P(downbeat class)
      └── downbeat = P(downbeat class)
                    │
                    ▼
          existing LiveTracker + BarTracker
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   tactus grid/BPM      bar phase + tactus grouping
                                │
                                └── optional tiny subdivision/meter-family head
                                    only after M0b
```

Default candidate зберігає `272` features, `2` LSTM layers, hidden size `150`, `3` softmax classes і всі tensor shapes. Це дозволяє повторно використати `TTBN v1`, current C++ inference та parity tests без нового runtime.

BeatNet+ не є drop-in replacement: у нього інший feature contract і чотири LSTM layers. Він може бути research comparator або окремою пізньою гілкою, але не повинен непомітно підміняти дешевий BeatNet fine-tune.

## 6. Обов'язкові preflight-рішення

### 6.1. Ліцензія і права

До дорогого training run зафіксувати:

- revision upstream BeatNet і точний digest source checkpoint;
- ліцензію коду, початкових ваг і похідних ваг;
- attribution та опис modifications;
- письмове/юридичне рішення щодо поширення CC BY 4.0 ваг у mobile stores з DRM;
- rights status кожного training recording.

Невирішена shipping license не забороняє окремий research benchmark, але всі його checkpoints і outputs мають бути позначені `research_only`. BeatNet+ лишається research-only, доки його code/weights license не буде явно зафіксована в manifest.

### 6.2. Upstream training source

Не копіювати trainer вручну. Закріпити конкретний commit офіційного [BeatNet repository](https://github.com/mjhydri/BeatNet), який містить training pipeline, і зберегти patch ledger для всіх локальних відмінностей.

Upstream defaults не приймаються автоматично:

- best checkpoint upstream обирається за beat F, а Tik-tak потребує ієрархічного beat/bar/meter gate;
- class weights, 8-second chunks і DBN validation є стартовими baselines, не затвердженими hyperparameters;
- `madmom` сумісність не повинна визначати production architecture; training environment ізолюється від core build.

### 6.3. Відтворюваний baseline

До fine-tuning відтворити pinned frozen BeatNet через:

1. upstream PyTorch inference;
2. локальний Python reference;
3. exported `TTBN v1` і C++ forward-pass;
4. той самий `LiveTracker` та `BarTracker`.

Frame activations, recurrent reset semantics і end-to-end events мають пройти preregistered parity tolerance. Інакше fine-tuning оптимізуватиме одну систему, а продукт запускатиме іншу.

## 7. Дані та split contract

### 7.1. Одиниця розділення

- Primary group — композиція.
- Performer, session, room, device та конкретне виконання не можуть перетікати між train/dev/locked через дублікати або paired captures.
- Nested learning-curve subsets формуються композиціями, а не уривками.
- Locked manifest недоступний training code і відкривається один раз.

### 7.2. Labels

Використати наявний `.beats` contract і versioned meter metadata; не створювати другий формат. На сегменті зберігати:

- beat times;
- downbeat times / beat position `1`;
- bar position кожної долі;
- tactus-beats-per-bar/grouping;
- subdivisions-per-tactus і `meter_family`;
- optional canonical numerator/denominator разом із `notation_basis`, без приписування denominator моделі;
- точний beat index meter change;
- `unknown/ambiguous` masks і annotation confidence.

Відсутній label маскується в loss; він не стає негативним прикладом. Beat-only public data може навчати beat evidence, але не downbeat, phase або meter.

### 7.3. Domain data

Primary adaptation data — реальні paired clean↔microphone записи після alignment QC. Pilot і sampling matrix беруться з [research/eval/LIVE_MIC_PILOT.md](research/eval/LIVE_MIC_PILOT.md) після синхронізації з основним планом.

Обов'язкові страти:

- room, distance, device, level, ensemble;
- playback off та реальний app click на кількох рівнях;
- 2/4, 3/4, 4/4, 6/8 і доступні складні/змінні розміри;
- pickup, syncopation, quiet/missing downbeats, drumless material, rubato, stops/restarts.

Synthetic RIR/noise/codec/AGC є лише candidate augmentations. Кожна приймається окремим ablation лише тоді, коли покращує реальні room captures без clean regression.

### 7.4. Три різні data/budget контури

| Набір/бюджет | Призначення | Заборона |
|---|---|---|
| `P1-B0` protocol pilot | 20–30 виконань: capture/alignment, дисперсія, annotation cost/QC, фізичний click bleed | не навчає модель і не дає learning curve |
| `P1-B1` train/dev corpus | composition-grouped `25/50/100%` learning curve, не менше 3 seeds, A5–A7 | не перетинається з locked |
| `locked` | незалежний one-shot confirmation | не доступний training code і не стає validation |
| Annotation/QC budget | людино-години первинної розмітки, подвійної перевірки, adjudication і rework | не підміняється кількістю записів |

До відкриття `P1-B1` за результатами B0 фіксуються: хвилини анотатора на хвилину аудіо; частка незалежної подвійної розмітки; reference-channel multiplier; правила adjudication; QC/rework rate; максимальні людино-години; обов'язкові поля tactus/downbeat/bar position/grouping/subdivision/`meter_family`/meter change/`notation_basis`/ambiguity. Якщо бюджет не сходиться, явно звужується label/product contract, а не мовчки послаблюється QC.

Наявні п'ять Harmonix pairs використовуються лише з leave-one-composition-out і лише як directional evidence. Для раннього click-bleed micro-check клік має фізично відтворюватися динаміком і заново захоплюватися мікрофоном; software mix не моделює room feedback, AEC або самопідтверджувальну петлю. Micro-check може виправити протокол до B0, але не замінює B0 і не дає продуктового висновку.

## 8. Training infrastructure

Створювати лише integration layer, якого немає upstream:

```text
research/training/beatnet/
  config + pinned upstream revision
  Tik-tak dataset adapter
  group split/leak audit
  masked label construction
  training launcher/checkpoint manifest
  product-metric validation callback
```

PyTorch і training-only packages належать в окреме optional environment; вони не додаються до core або shipping dependency graph.

Мінімальні вимоги:

- deterministic seeds і config snapshot;
- resume equivalence;
- tiny-set overfit для beat-only, beat+downbeat і full-meter batches;
- finite loss/gradient checks;
- loss masks для відсутніх labels;
- clip sampling без домінування 4/4 і non-beat frames;
- checkpoint selection за ієрархічним development score, а не лише beat F;
- exact data/model/code digests у кожному result.

## 9. Матриця адаптації

Усі arms отримують однакові train/dev splits, frame labels, decoder, latency budget, seed set та evaluation code.

| Arm | Що навчається | Сумісність із current core | Питання |
|---|---|---|---|
| A0 | Нічого: pinned frozen BeatNet | Повна | Абсолютний baseline |
| A1 | Лише calibration/decoder thresholds | Повна, ваги незмінні | Чи проблема взагалі потребує gradient update? |
| A2 | Наявний 3-class output layer | Повна | Чи достатньо перекалібрувати доменні класи? |
| A3 | Верхній LSTM layer + output layer | Повна | Чи потрібна temporal domain adaptation без руйнування frontend? |
| A4 | Уся наявна CRNN з discriminative LR | Повна | Максимум від тієї самої архітектури |
| A5 | Найкращий A2–A4 + supervised real room data | Повна | Чи закриває реальна кімната domain gap? |
| A6 | A5 + output consistency clean↔room | Повна | Чи додає paired invariance користь після supervised fine-tune? |
| A7 | A5/A6 + teacher logits лише на train data | Повна | Чи допомагає сильний offline teacher без test leakage? |

Порядок обов'язковий: `A0 → A1 → A2 → A3 → A4`; A5–A7 відкриваються після alignment і data-rights gates. Якщо A2 проходить gate, A3/A4 не потрібні. Якщо supervised A5 проходить gate, consistency/distillation не потрібні.

### Процедурні руки S0–S2

Ці руки ортогональні A-матриці: S0 нічого не навчає, S1 змінює процедуру навчання сумісної BeatNet, S2 є training-only допоміжною головою.

| Arm | Що перевіряє | Залежність |
|---|---|---|
| S0 | frozen BeatNet: reset LSTM-state кожні `2/4/8/16/32` с проти безперервного стану | немає; до S1 |
| S1 | stateful contiguous blocks, TBPTT `detach`, masked warm-up | лише позитивний S0; ablation поверх A2–A4 |
| S2 | training-only bar-position auxiliary head, вилучений перед export | лише після M0c і нового позитивного метричного гейта |

S0 ізолює **лише** стан `BeatNetModel`: feature history, `LiveTracker` і `BarTracker` не скидаються; однакові reset points не підбираються за музичними межами; transient після reset звітується окремо, а основна метрика рахується також із masked warm-up. Якщо continuous state не покращує downbeat/bar phase, S1 втрачає пріоритет. Якщо S0 позитивний, S1 не переносить state між композиціями або batch slots і робить reset лише там, де його причинно робить runtime.

Окрема tempo head/S3 не входить у матрицю: BPM уже працює, а темповий prior не дає потрібного метричного evidence.

### Loss contract для сумісної гілки

Базовий objective лишається 3-class frame classification. У продукті:

```text
p_any_beat = p_beat_class + p_downbeat_class
p_downbeat = p_downbeat_class
```

Тому `p_downbeat ≤ p_any_beat` виконується за конструкцією без нового coherence loss. Class weights або focal loss порівнюються як preregistered ablation; rare-class weighting не може змінювати unlabelled frames на negatives.

Paired consistency додається тільки між якісно вирівняними clean/room frames і не зсуває musical timestamp до delivery time. Спершу перевіряється output consistency; latent consistency відкривається лише якщо простіший objective не дає потрібного приросту.

## 10. Metrical ladder

Fine-tuning BeatNet безпосередньо навчає tactus/downbeat salience, а не `meter_family` чи нотний denominator. Щоб не приписати decoder failure мережі, використовується та сама чотирирука драбина, що в основному плані:

1. **Oracle decoder:** reference tactus + reference downbeats → current `BarTracker/resolveMeter`.
2. **Downbeat ceiling:** reference tactus + predicted downbeats → current decoder.
3. **Grid sufficiency:** predicted tactus + oracle phase → current decoder.
4. **Full frontend:** predicted tactus + predicted downbeats → current decoder.

Інтерпретація:

- якщо 1 не проходить bar phase/grouping, спершу виправляється decoder або label contract;
- якщо 1 проходить, а 2 ні, fine-tuning має покращувати downbeat salience/context;
- якщо 2 проходить, а 3 ні, вузьке місце у tactus grid;
- якщо phase/grouping проходять, але subdivision/`meter_family` ні, поточні BeatNet outputs і decoder інформаційно недостатні для цієї задачі.

`M0a` запускається одразу після provenance/parity на найметричнішому наявному матеріалі. Негатив oracle-руки 1 є твердим stop для neural escalation; позитив лише не спростовує її через домінування 4/4. `M0b` повторює драбину на meter-diverse dev. Фактичний M0b — `inconclusive`: статичні A1 phase/grouping високі, але швидке захоплення змін провалене, а 62/123 RWC2 переходів right-censored щодо повної двотактової межі. Тому перед будь-яким S2/metrical adapter виконується A1-only `M0c` transition trace; він локалізує stale state проти phase/sequence instability, але сам не є позитивним гейтом. Denominator у M0 не оцінюється, бо поточний decoder його не видає, а звук його однозначно не визначає.

Лише останній випадок після M0c і нового позитивного метричного гейта відкриває маленький **metrical adapter**, не новий audio frontend:

- перший baseline — deterministic subdivision evidence;
- другий — tiny beat-synchronous classifier на accepted tactus sequence;
- третій — small head на frozen/fine-tuned BeatNet hidden state, якщо beat sequence недостатньо;
- output завжди має calibrated `unknown`, risk–coverage і окремі false-confident/unnecessary-abstention rates;
- adapter порівнюється з `always 4`, empirical prior і current `BarTracker`; `canonical_time_signature` звітується окремо з `notation_basis`.

Зміна 3-class audio head на hierarchical/independent heads дозволена лише після окремого доказу, що сумісна softmax-гілка вичерпана. Вона потребує нового artifact format, exporter, C++ forward-pass і parity suite, тому вже не вважається дешевим fine-tune.

## 11. Preregistered gates

До перегляду comparative results зафіксувати числові thresholds у новому `research/eval/PREREGISTERED_beatnet_finetune.md`.

### Quality

- beat F/precision/recall, `usable_strict`, extra/missed beats, wrong-level duration, never-settled fraction, `beats_late`;
- downbeat F/precision/recall, bar-onset F, bar-phase accuracy/continuity, acquisition time;
- macro-F1/balanced accuracy для tactus grouping і `meter_family`; canonical time-signature classes звітуються вторинно за `notation_basis`;
- per-class coverage, risk–coverage, selective error, false-confident/unnecessary-abstention rates, meter-change F/latency;
- clean non-inferiority margin;
- minimum worthwhile room improvement з lower paired/group bootstrap bound.

### Cost and causality

- `TTBN v1` shape/size для всіх сумісних arms;
- unchanged 50-fps timing і recurrent reset semantics;
- parameter/MAC/state-size cap;
- static core build і allocation-free callback tests;
- P5 on-device RTF, RAM, startup, energy і thermal budgets.

Затримка завжди звітується розкладом: `32 ms` від центрованого frontend-вікна BeatNet ([beatnet.hpp:139](core/src/ml/beatnet.hpp#L139)) + explicit lookahead `L` + block/buffer delay + tracker/event delay. `L=0` означає нульове **додаткове** утримання, а не нульову total latency; musical timestamp не зсувається до delivery time.

### Hierarchical pass rule

Кандидат проходить лише якщо одночасно проходять beat, downbeat/bar-phase, meter/abstention, clean, room і mobile gates. Жодна середня або BPM не може перекрити провал нижчого рівня.

### Early futility rule

До comparative training окремо від final pass thresholds фіксуються мінімально корисний продуктовий приріст проти frozen A0 і CI rule для tactus-grid, downbeat/bar-phase та room. Його не прив'язують до `0.754` Beat This! на Harmonix або до довільної частки matched GTZAN gap. П'ять Harmonix pairs не можуть закрити альтернативу. Після першого достатньо потужного `P1-B1` зрізу та щонайменше трьох seeds arm можна зупинити, якщо верхня межа прогнозованого ефекту на запланованому data budget нижча за minimum worthwhile improvement. Крута learning curve або нестабільна оптимізація дають `inconclusive/data-limited`, а не архітектурний негатив.

## 12. Фази виконання

### F0 — доказова й юридична база

1. Закріпити upstream training revision і source checkpoint digest.
2. Вирішити research-only/shipping status похідних ваг.
3. Закрити provenance issues з основного плану й працювати в clean eval-worktree.
4. Відтворити PyTorch ↔ local Python ↔ `TTBN v1` ↔ C++ parity.
5. Зафіксувати frozen A0 на current dev suites.

**Exit:** один відтворюваний baseline artifact із повними model/code/data digests.

### F0a — безкоштовні ранні діагностики

1. Виконати M0a: чотири oracle/frontend руки на найметричнішому наявному матеріалі.
2. Виконати S0 reset/state-horizon diagnostic з ізоляцією лише LSTM-state.
3. Виконати A1 на п'яти Harmonix pairs із leave-one-composition-out як directional read.
4. Провести фізичний click-bleed micro-check, якщо пререєстрована M0-click розвилка лишає адаптацію під час кліку продуктовою вимогою.

**Exit:** M0a або твердо зупинив neural escalation, або не спростував її; S0 вирішив, чи відкривати S1; A1 і click micro-check вплинули лише на порядок та протокол, не на final gate.

### F1 — тонкий training adapter

1. Під'єднати наявні annotations/splits до pinned upstream trainer.
2. Додати masks, group sampler, leak audit і project metric callback.
3. Пройти tiny-set overfit, deterministic rerun і resume equivalence.
4. Перевірити export донавченого synthetic checkpoint через current exporter.

**Exit:** smoke training створює core-loadable `TTBN v1`, а parity suite проходить.

### F2 — protocol pilot, train/dev corpus і learning curve

1. `F2a/P1-B0`: зібрати 20–30 protocol-pilot виконань; перевірити capture/alignment, click bleed, дисперсію та реальну annotation cost. Навчання на B0 не запускати.
2. За B0 заморозити protocol, annotation/QC budget, split rules і sampling matrix.
3. `F2b/P1-B1`: зібрати окремий composition-grouped train/dev corpus.
4. Побудувати `25/50/100%` learning curve щонайменше на 3 seeds.
5. Окремо оцінити comparative power і per-meter/change coverage.

**Exit:** є окремі protocol і training висновки та оцінені data/annotation budgets. Steep `100%` slope не видається за saturation: або B1 розширюється, або продуктова обіцянка звужується; locked не відкривається і BeatNet не оголошується вичерпаним.

### F3 — дешева адаптація

1. Перенести directional A1 result із F0a; повторювати лише якщо змінився baseline/protocol.
2. A2 head-only.
3. A3 top-LSTM + head лише якщо A2 не проходить.
4. A4 full fine-tune лише якщо A3 не проходить.
5. S1 — лише після позитивного S0 і лише як ablation поверх A2–A4.

**Stop:** перший arm, що проходить усі preregistered dev gates, стає candidate; дорожчі arms не запускаються без окремої причини.

### F4 — room robustness

1. Supervised real-room A5.
2. Output consistency A6 лише після positive alignment audit.
3. Teacher logits A7 лише після simple-arm failure і corpus ledger audit.
4. Synthetic augmentations — одна за одною, з real-room acceptance.

**Exit:** lower CI room improvement проходить gate, clean non-inferiority збережена.

### F5 — metrical completion

1. M0b виконано: чотири oracle/frontend arms із §10 на meter-diverse dev;
   результат `inconclusive` через change acquisition.
2. Виконати пререєстрований M0c на A1 transitions, окремо від right-censored,
   і визначити домінуючий stale-state/phase/other механізм.
3. Перевірити найменший decoder counterfactual, який випливає з M0c; не
   переписувати M0b і не трактувати M0c як дозвіл на neural head.
4. Додати S2 або metrical adapter лише якщо наступний позитивний метричний гейт
   доведе subdivision/`meter_family` bottleneck.
5. Перевірити changes, acquisition, abstention і class balance.

**Exit:** bar position, grouping і `meter_family` проходять окремі gates; canonical time signature лишається вторинним виходом із `notation_basis`.

### F6 — candidate export і mobile

1. Додати fine-tuned artifact до manifest з source checkpoint, training config/data digests і modifications.
2. Конвертувати current exporter без зміни формату, якщо shapes незмінні.
3. PyTorch ↔ `TTBN` frame parity і end-to-end beat/downbeat/bar parity.
4. Виміряти actual mobile budgets на названих iOS/Android devices.
5. Quantization відкривати лише як окремий arm з quality regression gate.

### F7 — locked confirmation

1. Заморозити commit, checkpoint digest, config, seeds, thresholds і exclusion rules.
2. Перевірити locked meter-stratum coverage до відкриття.
3. Відкрити split один раз.
4. Негатив не ремонтувати tuning на тому самому split.

## 13. Failure modes і захист

| Failure mode | Захист |
|---|---|
| Fine-tune оптимізується лише за beat F | Hierarchical validation/checkpoint score з hard bar/meter floors |
| Поточний core запускає не ті weights | Source/training/export/runtime digests і frame parity |
| Head-only називають full fine-tune або навпаки | Manifest містить frozen/unfrozen parameter groups і optimizer config |
| 4/4 prior маскує meter failure | Meter-diverse splits, macro/balanced metrics, `always 4` baseline |
| Downbeat quality добра, а phase погана | Окремі downbeat і bar-phase метрики плюс oracle decoder |
| Denominator видано за акустичне передбачення | `meter_family` гейтиться окремо; canonical signature має `notation_basis` і не зараховує corpus/user convention моделі |
| B0 непомітно став training/dev | Явні `P1-B0`/`P1-B1`, training заборонений до завершення B0 analysis |
| P2 закрито на п'яти Harmonix pairs або steep curve | П'ять пар лише directional; futility rule застосовується лише на достатньо потужному B1 зрізі |
| Consistency loss навчається на поганому alignment | Alignment QC gate і shared musical timeline |
| Teacher переносить test knowledge | Corpus ledger; logits лише для train compositions |
| Synthetic room перемагає лише на synthetic test | Acceptance винятково на real paired room dev |
| Fine-tuning руйнує clean domain | Clean non-inferiority hard gate |
| Full fine-tune катастрофічно забуває pretraining | Discriminative LR, frozen baseline, layerwise arms, early stopping |
| BeatNet+ непомітно стає shipping dependency | Окремий artifact ID і research-only license gate |
| Зміна heads ламає дешевий deployment path | Несумісна branch відкривається лише після compatible-arm failure |
| Locked split стає validation | One-shot manifest і заборона post-locked tuning |

## 14. План перевірок

```text
pinned upstream + checkpoint
          │
          ▼
Tik-tak dataset adapter ──► train arm ──► checkpoint + manifest
          │                                  │
          ├── split/leak tests                ├── PyTorch frame metrics
          ├── label/mask tests                └── TTBN export + parity
          │                                              │
          ▼                                              ▼
meter-diverse dev ─────────────────────────► LiveTracker + BarTracker
                                                        │
                                      beat ── downbeat ── phase ── meter
                                                        │
                                                        ▼
                                                 mobile + locked
```

Обов'язкові test groups:

- data split/leak, label masks, meter changes і ambiguous regions;
- annotation timing, independent double-annotation, adjudication та QC/rework accounting;
- tiny overfit, deterministic repeat, resume equivalence;
- class order та `beat_total=beat+downbeat`;
- S0 isolation: reset лише `BeatNetModel`, незмінні feature/tracker states, fixed reset points і окремий transient;
- S1 state carry/detach, no cross-composition/batch-slot leakage, chunk-boundary і causality parity;
- Python/C++ frame and event parity;
- чотирирука M0a/M0b oracle/frontend ladder;
- real clean↔room paired evaluation;
- physical click-bleed capture; software-mixed click не зараховується;
- manifest/schema validation;
- on-device sustained run і no-allocation callback.

## 15. Implementation tasks

Кожен пункт — окремий reviewable change після окремого дозволу на виконання:

1. **BeatNet training pin** — upstream revision, license/provenance ledger, reproducible environment.
2. **Dataset adapter** — reuse `.beats`/meter metadata, masks і composition-grouped splits.
3. **Leak and rights audit** — composition/performance/session/room/device та `research_only` enforcement.
4. **Training smoke suite** — overfit, determinism, resume, finite gradients.
5. **Product-metric validation callback** — same `LiveTracker`/`BarTracker`, hierarchical checkpoint selection.
6. **Frozen baseline artifact** — verified A0 through Python and C++.
7. **M0a + S0 diagnostics** — four oracle/frontend arms; isolated reset/state-horizon sweep.
8. **B0 protocol and annotation report** — capture/alignment, physical click bleed, annotation/QC unit cost and budget.
9. **B1 learning curve and power report** — grouped `25/50/100%` subsets, ≥3 seeds, slope CI, meter/change coverage, futility projection.
10. **A1–A4 adaptation matrix** — S1 only after positive S0; stop at first full-gate pass.
11. **Paired room A5–A7 matrix** — supervised, output consistency, teacher only as justified.
12. **M0b metrical oracle report** — final meter-diverse decoder/downbeat/grid/full-frontend gate.
13. **Optional S2/metrical adapter** — only after M0c and a new positive metrical gate; `meter_family` and measured `unknown` behavior.
14. **Fine-tuned artifact/export** — manifest modifications, `TTBN v1` parity, C++ load.
15. **Mobile candidate report** — decomposed latency, RAM, startup, sustained RTF, energy/thermal.
16. **One-shot locked report** — immutable preregistration and final verdict.

Definition of done: кожен artifact містить commit, upstream revision, source checkpoint, trained checkpoint, export/runtime digests, data split/version, exact config/command, seeds, clean-tree status, test results і висновок, який не виходить за межі виміряного.

## 16. Stop/go рішення

```text
compatible BeatNet fine-tune passes all gates?
    yes ──► stop neural R&D; ship candidate after license/mobile/locked gates
    no
    │
    ▼
M0b says decoder or meter-family evidence is bottleneck?
    yes ──► decoder / tiny metrical adapter, BeatNet audio frontend unchanged
    no
    │
    ▼
BeatNet compatible arms exhausted with enough data and stable optimization?
    no ──► improve evidence/data/training, do not invent a new model yet
    yes ──► close this alternative with failure report
             │
             └──► separate decision whether to execute own-model plan.md
```

Цей документ вважається успішним і тоді, коли доводить, що BeatNet не дотягує: цінність полягає в тому, що провал локалізований між data, frontend, beat decoder і metrical decoder, а перехід до власної моделі має конкретну причину, а не загальне враження.
