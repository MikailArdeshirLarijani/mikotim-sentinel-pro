# MikoTim Sentinel Pro

سامانه سبک پایش وب‌کم، تشخیص حرکت، ثبت چهره و ارسال هشدار به تلگرام  
Lightweight webcam monitoring system with motion detection, face registration, and Telegram alerts.

---

### 🌐 انتخاب زبان / Select Language
> برای مشاهده یا بستن هر زبان، روی عنوان آن کلیک کنید:  
> Click on the section title below to expand or collapse:

---

<details open>
<summary><h2>🇮🇷 توضیحات فارسی</h2></summary>

### درباره پروژه
این پروژه یک صفحه وب تک‌فایلی (`index.html`) است که با استفاده از جاوااسکریپت ساده و کانواس (Canvas)، وب‌کم سیستم را به یک دوربین پایش امنیتی تبدیل می‌کند. تمام پردازش‌های تصویر در مرورگر انجام می‌شود و هیچ اطلاعاتی به سرورهای خارجی ارسال نمی‌شود.

### قابلیت‌های واقعی سامانه
- **پایش تصویر با وب‌کم:** تشخیص حرکت در کادر دوربین با مقایسه فریم‌ها در کانواس مرورگر.
- **ثبت و تشخیص چهره:**
  - امکان ثبت چهره افراد به همراه نام آن‌ها در حافظه مرورگر (`localStorage`).
  - محاسبه یک بردار ویژگی ۴۲ نقطه‌ای (هیستوگرام رنگ و الگوی روشنایی).
  - مقایسه افراد حاضر در کادر با چهره‌های ثبت‌شده (تشابه کسینوسی).
  - کادر سبز به همراه نام برای فرد شناسایی‌شده و کادر زرد برای فرد ناشناس.
- **ارسال عکس به تلگرام:**
  - ارسال خودکار تصویر با کادر و زمان به چت تلگرام از طریق Bot API رسمی تلگرام.
  - گزینه «فقط هنگام ورود فرد جدید به کادر عکس بگیر».
- **کنترل از راه دور با دستورات تلگرام:**
  - `/photo` یا `عکس`: گرفتن عکس فوری از وب‌کم و ارسال به تلگرام.
  - `/status` یا `وضعیت`: گزارش وضعیت فعال یا غیرفعال بودن دوربین.
  - `/pause` یا `توقف`: توقف موقت ثبت رویدادها.
  - `/resume` یا `ادامه`: ادامه پایش.
- **رسم محدوده حساس (ROI):** قابلیت کشیدن کادر روی تصویر برای پایش فقط یک بخش خاص.
- **شیفت زمانی:** تعیین بازه ساعتی مشخص برای فعال بودن سیستم.
- **خروجی اکسل:** دانلود تاریخچه ترددها در قالب فایل CSV.
- **پخش صدا:** آلارم صوتی کوتاه هنگام تشخیص حضور فرد.
- **دوزبانه و تم:** امکان انتخاب زبان فارسی/انگلیسی و تم تیره/روشن.

### نحوه اجرا
1. فایل `index.html` را در مرورگر (مانند Chrome یا Edge) باز کنید.
2. دکمه **«شروع پایش»** را بزنید و اجازه دسترسی به وب‌کم را تایید کنید.

### تنظیم اتصال به تلگرام
1. در تلگرام به ربات `@BotFather` رفته و با دستور `/newbot` یک ربات بسازید و توکن آن را کپی کنید.
2. شناسه عددی اکانت خود را از ربات `@userinfobot` دریافت کنید.
3. در تب **تنظیمات** برنامه:
   - توکن ربات و شناسه چت را وارد کنید.
   - دکمه **تست اتصال** و سپس **ذخیره تنظیمات** را بزنید.

</details>

---

<details open>
<summary><h2>🇬🇧 English Documentation</h2></summary>

### Overview
MikoTim Sentinel Pro is a single-file web application (`index.html`) built with vanilla JavaScript and HTML5 Canvas. It turns your webcam into an offline motion and security monitor that can register faces and dispatch photo alerts to Telegram. All video processing runs locally in your browser.

### Real Features
- **Webcam Monitoring:** Real-time motion detection using browser Canvas pixel differencing.
- **Face Registration & Recognition:**
  - Register individuals with their name, saved locally in browser `localStorage`.
  - Extracts a 42-point feature signature (color histogram and spatial grid pattern).
  - Matches live faces against saved entries using cosine similarity.
  - Displays a green bracket with the person's name for matches, or an amber bracket for unknown subjects.
- **Telegram Photo Alerts:**
  - Automatically dispatches captured photos with bounding box and timestamp to your Telegram chat via official Bot API.
  - Optional filter: "Capture only when a new person enters".
- **Remote Telegram Commands:**
  - `/photo`: Takes an instant camera snapshot and sends it back to Telegram.
  - `/status`: Returns current camera and monitoring status.
  - `/pause`: Pauses detection and capture triggers.
  - `/resume`: Resumes normal monitoring.
- **Sensitive ROI Zone:** Draw a custom rectangle on the camera stream to monitor only a specific area.
- **Shift Scheduling:** Set active hours (e.g. night shift) to restrict alerts to specific times.
- **CSV Export:** Download the incident history as a CSV spreadsheet.
- **Audio Chime:** Plays a short chime through Web Audio API upon subject detection.
- **Bilingual & Theme:** Built-in English / Persian language switcher and Dark / Light theme.

### How to Run
1. Open `index.html` directly in any modern browser (Chrome, Edge, Firefox, Brave).
2. Click **Start Monitoring** and grant camera permissions.

### Telegram Setup
1. Create a bot using `@BotFather` on Telegram and copy the Bot Token.
2. Get your numeric Chat ID using `@userinfobot`.
3. In the application's **Settings** tab:
   - Enter your Bot Token and Chat ID.
   - Click **Test Connection**, then click **Save Settings**.

</details>

---

### 📄 مجوز / License
[MIT License](LICENSE)
