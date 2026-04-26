# Visual_Novel_Studio
StoryWeaver - Безкоштовний Редактор Візуальних Новел
Детальна дорожна карта розробки

🎯 ПРОЕКТНА СТРАТЕГІЯ
Видіння
Створити найпростіший у світі мобільний редактор, де будь-хто (без коду) за 30 хвилин може написати першу сцену своєї історії.
Цільова аудиторія

Автори без технічного досвіду
Письменники які хочуть інтерактивні творіння
Викладачі для освітніх проектів
Новачки у game development

USP (Унікальна пропозиція)

100% мобільний (создаточ, редагування, експорт)
Offline-first — працює без інтернету
Zero Code — синтаксис для людей, не для programmers
Cross-platform — один код для iOS/Android/Web


📦 АРХІТЕКТУРА ПРОЕКТУ
Tech Stack
┌─────────────────────────────────────────────┐
│          StoryWeaver Platform               │
├─────────────────────────────────────────────┤
│                                             │
│  FRONTEND (Expo/React Native)              │
│  ├─ Mobile App (iOS/Android)               │
│  ├─ Web Editor (React)                     │
│  └─ Desktop App (Electron - факультативно)│
│                                             │
│  STORAGE                                    │
│  ├─ SQLite (локально на пристрої)          │
│  ├─ FileSystem API (для медіа)             │
│  └─ Cloud Sync (Firebase/Supabase)         │
│                                             │
│  BACKEND (Node.js)                         │
│  ├─ User Management                        │
│  ├─ Cloud Backup Service                   │
│  ├─ Export Service (HTML/APK)              │
│  └─ Analytics                              │
│                                             │
│  DEPLOYMENT                                 │
│  ├─ App Store / Google Play                │
│  ├─ Web (Vercel/Netlify)                   │
│  └─ Self-hosted option                     │
│                                             │
└─────────────────────────────────────────────┘
Структура репозиторію
storyweaver/
├── packages/
│   ├── mobile/              # React Native (Expo)
│   │   ├── src/
│   │   │   ├── screens/
│   │   │   ├── components/
│   │   │   ├── services/
│   │   │   └── database/
│   │   └── app.json
│   │
│   ├── web-editor/          # React Web
│   │   ├── src/
│   │   │   ├── pages/
│   │   │   ├── components/
│   │   │   ├── hooks/
│   │   │   └── utils/
│   │   └── vite.config.js
│   │
│   ├── core/                # Shared Logic
│   │   ├── parser.js        # Синтаксис парсер
│   │   ├── validator.js
│   │   ├── compiler.js
│   │   └── types.ts
│   │
│   └── backend/             # Node.js
│       ├── src/
│       │   ├── routes/
│       │   ├── controllers/
│       │   ├── middleware/
│       │   └── models/
│       └── package.json
│
├── docs/
│   ├── SYNTAX.md            # Language specification
│   ├── ARCHITECTURE.md
│   ├── API.md
│   └── DEPLOYMENT.md
│
└── package.json (monorepo)

🗂️ СЦЕНАРІЙНА МОВА
Синтаксис (простий, інтуїтивний)
; Коментарій

[Сцена 1: Назва сцени]
; Команди сцени
@bg BG_NAME fade:1.0           ; Фон з переходом
@music MusicTrack loop:true    ; Музика
@char Hero enter left           ; Персонаж входить зліва
@char Leika enter right         ; Персонаж входить справа

; Діалог (автор: текст)
Hero: "Привіт, Лейка!"
Leika: "Привіт, Герой!"

; Розповідь (без персонажа)
Це спокійний ранок у Факторі Знань.

; Вибір
@choice
- "Привітатися" #greeting
- "Проходити повз" #walk_by
- "Спитати про день" #ask

; Мітка
# greeting
Hero: "Як справи?"
@goto #end

# walk_by
Hero: *ходить далі*
@goto #end

# ask
Hero: "Як твій день?"
Leika: "Прекрасно! А твій?"
@goto #end

# end
Це була цікава зустріч.
@clear
Комбіновані функції
; Змінні
@set player.name = "Герой"
@set player.mood = happy
@set stats.chapters = 1

; Умови
@if player.mood == happy
    Hero: "Я в чудовому настрої!"
@endif

@if stats.chapters > 3
    Hero: "Ми набагато далі в історії..."
@else
    Hero: "Ми тільки на початку..."
@endif

; Ефекти
@effect shakeScreen duration:0.5 intensity:10
@effect fadeIn duration:1.0
@effect glitch duration:0.3

; Звуки
@sfx ButtonClick volume:0.8
@voiceActing Hero greet_001.ogg

; Інтеграція з медіа
@image CharacterPortrait show:true duration:0.5

🚀 ФАЗА-ЗА-ФАЗОЮ РОЗРОБКА
ФАЗА 1: MVP Основа (Тижні 1–4)
Цілі:

✅ Базовий проект manager
✅ Простий текстовий редактор
✅ Parser для синтаксису
✅ Basic preview/player

Deliverables:

Home Screen (storyweaver-home.jsx)

Проекти список
-Create/Delete/Duplicate
Last edited tracking


Script Editor (script-editor.jsx)

Textarea з текстом
Character management
Basic syntax highlighting
Line numbering


VN Player (vn-player.jsx)

Dialogue display
Character positioning
Simple choices
Progress indicator


Core Parser (parser.js)

javascript   // Парсить сценарій в AST
   const ast = parseScript(scriptText);
   // Output:
   // [
   //   { type: 'scene', name: 'Введення', ... },
   //   { type: 'command', action: 'bg', target: 'BG_NAME', ... },
   //   { type: 'dialogue', speaker: 'Hero', text: '...' },
   //   ...
   // ]

Database Schema (SQLite)

sql   -- Projects
   CREATE TABLE projects (
     id TEXT PRIMARY KEY,
     name TEXT,
     description TEXT,
     scriptContent TEXT,
     characters JSON,
     createdAt DATETIME,
     lastEditedAt DATETIME,
     status TEXT
   );

   -- Assets (images, music, etc)
   CREATE TABLE assets (
     id TEXT PRIMARY KEY,
     projectId TEXT,
     type TEXT, -- image, audio, video
     name TEXT,
     filePath TEXT,
     uploadedAt DATETIME
   );
Timeline: 4 тижні
Resources: 1 Frontend Dev + 1 Backend Dev

ФАЗА 2: Контент & Розширення (Тижні 5–8)
Цілі:

✅ Asset Manager (зображення, музика)
✅ Розширена синтаксис (змінні, умови)
✅ Персонаж експресії/позиції
✅ Музичне керування

Features:

Asset Manager

Upload image / audio
Gallery view
Delete / rename
Preview


Character System

Create character profiles
Color for dialogue
Expression system (happy, sad, angry, etc)
Position on screen (left, center, right)


Advanced Script Commands

   @set variable = value
   @if condition @then action @endif
   @bg ImageName fade:duration
   @music TrackName loop:true
   @effect effectName param:value
   @char CharName enter|exit position

UI Improvements

Syntax highlighting
Line numbers
Bracket matching
Auto-complete



Timeline: 4 тижні
Resources: 2 Frontend Devs + 1 Backend Dev

ФАЗА 3: Полірування & Експорт (Тижні 9–12)
Цілі:

✅ Експорт (HTML/APK/iOS)
✅ UI/UX Polish
✅ Performance optimization
✅ Multiplayer sync (опціонально)

Features:

Export System

HTML Export: Standalone .html file
APK Export: Through EAS Build (Expo)
Web Publish: Deploy to GitHub Pages / Vercel



javascript   // Export API
   const exportOptions = {
     format: 'html', // 'html' | 'apk' | 'web'
     includeAssets: true,
     theme: 'dark', // 'light' | 'dark' | 'custom'
   };
   await exportProject(project, exportOptions);

Cloud Sync

Firebase Realtime Database
Auto-backup to cloud
Version history
Conflict resolution


Settings

Theme (light/dark)
Font size
Auto-save interval
Backup preferences


Performance

Lazy loading assets
Memory optimization
Script compilation caching
Image compression



Timeline: 4 тижні
Resources: 2 Devs + 1 QA

ФАЗА 4: Community & Distribution (Тижні 13–18)
Цілі:

✅ App Store / Play Store submission
✅ Documentation & Tutorials
✅ Community features
✅ Analytics

Features:

Distribution

App Store review prep
Play Store submission
Website launch
Tutorial videos


Community

Template library
Sharing system
Rating/Reviews
Discord community


Documentation

User guide (українська)
Video tutorials
Example projects
API documentation


Analytics

Usage metrics
User feedback
Crash reporting
Feature usage tracking



Timeline: 6 тижні
Resources: 1 PM + 1 Marketing + Full team

💾 БАЗА ДАНИХ ДИЗАЙН
Entities
javascript// Project
{
  id: string,
  name: string,
  description: string,
  thumbnail: string, // base64 або path
  characters: Character[],
  scenes: Scene[],
  assets: Asset[],
  variables: Variable[],
  scriptContent: string,
  status: 'draft' | 'published',
  createdAt: timestamp,
  lastEditedAt: timestamp,
  userId: string,
  isPublic: boolean
}

// Character
{
  id: string,
  name: string,
  color: string, // hex color
  expressions: Expression[],
  defaultPosition: 'left' | 'center' | 'right',
  voiceActing?: string // audio file id
}

// Expression
{
  id: string,
  name: string, // 'happy', 'sad', etc
  emoji?: string,
  imageId?: string
}

// Asset
{
  id: string,
  projectId: string,
  type: 'image' | 'audio' | 'video',
  name: string,
  filePath: string,
  metadata: {
    size: number,
    duration?: number,
    dimensions?: { width, height }
  }
}

// Variable
{
  id: string,
  name: string,
  type: 'string' | 'number' | 'boolean',
  defaultValue: any,
  isVisible: boolean
}

🎨 ДИЗАЙН СИСТЕМА
Colors
Primary: #7c3aed (Purple)
Secondary: #ec4899 (Pink)
Accent: #f59e0b (Amber)
Background: #f8fafc (Light)
Dark BG: #0f172a (Dark)
Text: #1e293b (Dark Text)
Muted: #64748b (Gray)
Typography
Display: 28px, 700 weight
Heading: 20px, 700 weight
Body: 16px, 500 weight
Caption: 12px, 400 weight
Monospace: 14px, 400 weight (code)
Components

Button (primary, secondary, ghost)
Input (text, textarea)
Card (project card, character card)
Modal (create, edit, delete)
Tabs
Slider / Range
Color Picker


📱 RESPONSIVE DESIGN
Mobile: 320px - 768px
  - Single column layout
  - Bottom navigation
  - Full-width cards
  - Larger touch targets

Tablet: 768px - 1024px
  - Two column layout
  - Sidebar + Main
  - Optimized for landscape

Desktop: 1024px+
  - Three panel layout
  - Sidebar + Editor + Preview
  - Advanced features enabled

🧪 TESTING STRATEGY
Unit Tests
javascript// parser.test.js
describe('Script Parser', () => {
  test('parses simple dialogue', () => {
    const script = `Hero: "Hello!"`;
    const ast = parseScript(script);
    expect(ast[0].type).toBe('dialogue');
    expect(ast[0].speaker).toBe('Hero');
  });

  test('parses choices', () => {
    const script = `@choice\n- "Option 1"\n- "Option 2"`;
    const ast = parseScript(script);
    expect(ast[0].type).toBe('choice');
    expect(ast[0].options.length).toBe(2);
  });
});
Integration Tests

E2E с headless browser
Script execution
Asset loading
Export functionality

Performance Tests

Parse speed (< 100ms for 50 scenes)
Memory usage (< 50MB for project)
Asset loading (parallel downloads)


🔐 Security & Privacy
Data Protection

Encryption at rest (SQLite with SQLCipher)
Encryption in transit (HTTPS)
User authentication (JWT)
GDPR compliance

Permissions

Camera (for character images)
Microphone (for voice acting)
Storage (for assets)
Network (for cloud sync)


📊 METRICS & KPIs
Success Criteria

5,000+ downloads in 3 months
4.5+ star rating
50%+ daily active users
< 2% crash rate
< 3 second load time

Analytics Tracking

User acquisition source
Feature usage
Session duration
Churn rate
Export/publish actions


🚢 DEPLOYMENT CHECKLIST
Pre-Launch

 All tests passing (>90% coverage)
 Performance optimized (< 10MB app size)
 UI/UX tested on real devices
 Privacy policy & ToS drafted
 App Store/Play Store accounts created

Launch

 Beta testers feedback incorporated
 App Store submission
 Play Store submission
 Website launch
 Social media announcement
 Discord community setup

Post-Launch

 Monitor crash reports
 User feedback analysis
 Performance monitoring
 Weekly updates
 Community engagement


💡 FUTURE FEATURES (Post-MVP)

Advanced Audio

Voice acting system
Background music management
Sound effect library


Visual Effects

Particle effects
Screen transitions
Camera movements
Video backgrounds


Multiplayer

Collaborative editing
Shared projects
User comments
Version control


Analytics

Player heatmaps
Choice statistics
Reading time tracking
User flow analysis


Monetization

Premium themes
Asset pack subscriptions
Publishing revenue share
Advanced export


AI Features

Story outline generation
Character name suggestions
Dialogue auto-complete
Image generation
