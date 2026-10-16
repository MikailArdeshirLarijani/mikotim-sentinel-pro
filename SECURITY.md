# Security Policy 🔒

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 4.x.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

Security is a core pillar of MikoTim Sentinel Pro. Because this software is designed for physical security and monitoring:

1. **Local Privacy**: All optical vision processing and face biometric analysis execute 100% locally in the browser memory. No video streams or captured frames are ever sent to external cloud servers.
2. **Telegram Credentials**: Bot tokens and Chat IDs are stored exclusively in your browser's local storage (`localStorage`). They are only sent over HTTPS directly to Telegram's official API (`https://api.telegram.org`).

If you discover a security vulnerability, please open a private security advisory on GitHub or email the maintainers directly. Please do not publish security issues publicly until they have been addressed.
