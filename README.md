# MikoTim Sentinel Pro

سامانه متن‌باز پایش و نظارت تصویری هوشمند با پایتون، YOLO و تلگرام  
**Open-source intelligent surveillance & identity monitor using Python, YOLO, and Telegram**

---

## معرفی پروژه | Overview

### فارسی
**MikoTim Sentinel Pro** یک ابزار محلی و متن‌باز برای نظارت تصویری و هشدار امنیتی است. این سیستم به وسیله وب‌کم یا دوربین متصل به سیستم، محیط را پایش می‌کند و در صورت حضور افراد:
- تردد فرد را به کمک شبکه عصبی **YOLOv8** به صورت دقیق تشخیص می‌دهد.
- چهره فرد را با پروفایل‌های ثبت‌شده در پایگاه داده محلی مقایسه می‌کند.
- تصویر فریم و گزارش رخداد را مستقیماً به چت یا کانال تلگرام ارسال می‌کند.
- امکان کنترل و دستوردهی از راه دور از طریق تلگرام (مانند دریافت عکس زنده یا توقف موقت) را فراهم می‌سازد.
- تمام پردازش‌ها ۱۰۰٪ روی سیستم کاربر انجام شده و هیچ داده‌ای به سرورهای خارجی ارسال نمی‌شود.

### English
**MikoTim Sentinel Pro** is a lightweight, local video surveillance tool. Connected to any standard webcam, it continuously monitors an area and:
- Detects human presence using an optimized **YOLOv8** model.
- Matches detected faces against saved profiles using normalized feature embeddings and cosine similarity.
- Sends instant incident photos and metadata to your Telegram chat.
- Accepts remote Telegram commands (`/photo`, `/status`, `/pause`, `/resume`).
- Runs completely offline on your local machine with zero external cloud dependencies.

---

## قابلیت‌های کلیدی | Key Features

- **تشخیص اشخاص با یادگیری عمیق (Deep Learning):** برخلاف روش‌های سنتی تفاضل فریم که با هر تغییر نوری آلارم خطا می‌دادند، مدل YOLO مستقیماً شکل و ساختار بدن انسان را شناسایی می‌کند.
- **بانک چهره و احراز هویت محلی:** ثبت سریع پروفایل افراد مجاز از روی تصویر زنده و شناسایی نام آن‌ها روی فریم.
- **پنل وب دوزبانه (فارسی / انگلیسی):** رابط کاربری واکنش‌گرا با پشتیبانی از تم تاریک و روشن، استریم زنده MJPEG و جدول لاگ وقایع.
- **یکپارچه‌سازی کامل با تلگرام:** ارسال گزارش هشدار با عکس، تاریخ و ساعت، درصد اطمینان و نام فرد شناسایی‌شده.
- **تنظیمات ذخیره‌شونده:** مدیریت زمان تاخیر بین هشدارها (Cooldown)، فیلتر عدم تکرار هشدار برای فرد قبلی و تنظیم شیفت‌های کاری.

---

## نیازمندی‌ها و نصب | Requirements & Setup

### نیازمندی‌ها | Prerequisites
- پایتون نسخه 3.10 یا بالاتر
- یک وب‌کم یا دوربین USB

### ۱. کلون کردن مخزن و نصب وابستگی‌ها | Clone & Install
```bash
git clone https://github.com/MikailArdeshirLarijani/mikotim-sentinel-pro.git
cd mikotim-sentinel-pro
pip install -r requirements.txt
```

### ۲. اجرای سامانه | Running the App
می‌توانید فایل `run.bat` را اجرا کنید یا دستور زیر را در ترمینال بزنید:
```bash
python app.py
```
سپس مرورگر خود را باز کرده و به آدرس زیر بروید:
👉 **http://127.0.0.1:5000**

---

## اجرای تست‌های خودکار | Running Tests
پروژه شامل یک مجموعه تست واحد برای بررسی عملکرد پایگاه داده چهره، ریاضیات بردارها، درگاه تلگرام و مسیرهای سرور است:
```bash
python -m unittest tests/test_suite.py
```

---

## ساختار فایل‌ها | Directory Structure
```text
├── app.py              # سرور اصلی Flask و کنترل‌کننده‌های API
├── detector.py         # پایپ‌لاین تشخیص YOLO و استخراج ویژگی‌های چهره
├── telegram_bot.py     # کلاس ارتباط با تلگرام و پولینگ دستورات
├── templates/
│   └── index.html      # قالب وب دوزبانه و داشبورد زنده
├── tests/
│   └── test_suite.py   # تست‌های خودکار
├── faces/              # محل ذخیره بردارهای مشخصه چهره‌ها
├── requirements.txt    # لیست کتابخانه‌های پایتون
└── run.bat             # اسکریپت اجرای سریع در ویندوز
```

---

## لایسنس | License
این پروژه به صورت متن‌باز تحت پروانه [MIT](LICENSE) منتشر شده است.
