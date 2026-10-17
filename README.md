# MikoTim Sentinel Pro

سامانه متن‌باز پایش و نظارت تصویری هوشمند با پایتون، YOLO و تلگرام  
**Open-source intelligent surveillance & identity monitor using Python, YOLO, and Telegram**

---

## معرفی پروژه | Overview

### فارسی
**MikoTim Sentinel Pro** یک ابزار مدرن، محلی و متن‌باز برای نظارت تصویری هوشمند است. این سیستم قابلیت کارکرد به دو روش را دارد:
1. **حالت مستقل در مرورگر (Standalone Web):** بدون نیاز به هیچ پیش‌نیاز یا اجرای پایتون، صرفاً با باز کردن `index.html`، وب‌کم فعال شده و با تشخیص حرکت هشدارها را مستقیماً به ربات تلگرام ارسال می‌کند.
2. **حالت پردازش هوش مصنوعی با پایتون (AI Server):** اجرای `server.py` که پایپ‌لاین تشخیص اشخاص **YOLOv8** و تطبیق مشخصه‌های چهره را با استریم باکیفیت و کنترل کامل فعال می‌سازد.

- بدون ارسال اطلاعات به سرورهای خارجی (۱۰۰٪ امن و محلی).
- مجهز به رابط کاربری شیشه‌ای فوق‌مدرن (Apple-grade Glassmorphism) و دوزبانه (فارسی / انگلیسی).
- کنترل از راه دور از طریق تلگرام با دستورات متنی.

### English
**MikoTim Sentinel Pro** is a modern, lightweight, privacy-first surveillance system designed for instant deployment:
1. **Direct In-Browser Standalone:** Simply double-click `index.html`. It runs directly via WebRTC camera APIs and client-side motion tracking with direct Telegram dispatch—no server setup required.
2. **Deep Learning AI Server:** Launch `server.py` for full **YOLOv8** pose/human detection, normalized facial feature embeddings, and MJPEG real-time streaming.

- 100% offline and local execution; zero external cloud tracking.
- Sleek Apple-inspired glassmorphism design with dual language support (Persian / English).
- Remote Telegram management and snapshot alerts.

---

## ساختار ساده و یکپارچه | Project Structure

پروژه به صورت کاملاً ساختاریافته در ۵ فایل اصلی سازمان‌دهی شده است:

```text
mikotim-sentinel-pro/
├── index.html          # رابط کاربری مستقل و مدرن، پایش وب‌کم و سیستم اعلان تلگرام
├── server.py           # سرور یکپارچه پایتون (هوش مصنوعی YOLOv8، بانک چهره، تلگرام و وب)
├── requirements.txt    # پیش‌نیازها و وابستگی‌های پایتون
├── README.md           # مستندات و راهنمای جامع فارسی و انگلیسی
└── LICENSE             # پروانه انتشار متن‌باز MIT
```

---

## نحوه اجرا | Getting Started

### روش اول: اجرای مستقیم بدون نیاز به پایتون (Direct In-Browser)
فقط کافیست فایل `index.html` را در هر مرورگری (Chrome, Edge, Firefox) باز کنید. سیستم دوربین شما را باز کرده و بلافاصله آماده پایش و ارسال هشدار به تلگرام خواهد بود.

### روش دوم: اجرای سرور هوش مصنوعی پایتون (Python YOLOv8 Server)
در صورتی که می‌خواهید از پردازش عمیق YOLOv8 و دیتابیس برداری چهره استفاده کنید:

```bash
# ۱. نصب پیش‌نیازها
pip install -r requirements.txt

# ۲. اجرای سرور
python server.py
```

سپس مرورگر خود را باز کرده و وارد آدرس زیر شوید:  
👉 **http://127.0.0.1:5000**

---

## قابلیت‌های کلیدی | Key Features

- **طراحی سبک و تک‌فایلی بخش‌ها:** کاهش تمام پراکندگی‌های کدهای قدیمی به یک فایل کلاینت و یک فایل سرور سازمان‌یافته.
- **شبکه عصبی YOLOv8:** پایش مقاوم در برابر خطای نوری و تشخیص آنی ساختار بدن انسان.
- **بانک چهره و احراز هویت محلی:** ذخیره و تطبیق بردارهای هندسی چهره با الگوریتم شباهت کسینوسی.
- **دستوردهی از راه دور تلگرام:** پشتیبانی از دستوراتی مانند `/photo`, `/status`, `/pause`, `/resume`.
- **تنظیمات داینامیک:** تعیین زمان بازدارنده بین آلارم‌ها (Cooldown)، فیلتر جلوگیری از ثبت مجدد و تنظیمات دلخواه پیام.

---

## لایسنس | License
این پروژه به صورت متن‌باز تحت پروانه [MIT](LICENSE) منتشر شده است.
