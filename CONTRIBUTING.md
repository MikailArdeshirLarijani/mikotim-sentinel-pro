# Contributing to MikoTim Sentinel Pro 🛡️

Thank you for your interest in contributing to **MikoTim Sentinel Pro**! We welcome all contributions, whether it's bug reports, feature suggestions, documentation improvements, or pull requests.

## How Can I Contribute?

### 1. Reporting Bugs
- Search existing issues to ensure the bug hasn't been reported yet.
- Provide clear steps to reproduce the issue, your browser name/version, and any error messages from the browser console (F12).

### 2. Suggesting Enhancements
- Open a feature request issue.
- Clearly describe the problem you want solved and proposed solutions.
- Keep in mind the project philosophy: **100% offline, privacy-first, zero cloud dependencies, single-file distribution**.

### 3. Submitting Pull Requests
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Make your changes in `index.html`.
4. Ensure the system works offline without any external network calls (except user-configured Telegram Bot API).
5. Test bilingual compatibility (English and Persian RTL/LTR).
6. Commit your changes: `git commit -m "feat: add support for XYZ"`.
7. Push to your fork: `git push origin feature/your-feature-name`.
8. Open a Pull Request with a clear description of your changes.

## Code Standards
- **Single-File Architecture**: The entire application is self-contained in `index.html`.
- **Pure JavaScript**: No external frontend frameworks (React, Vue, etc.) or heavy WASM/CDN bundles.
- **Privacy & Security**: Zero telemetry, zero tracking, no third-party analytics.
- **Bilingual i18n**: All UI text must be registered in both `i18n.fa` and `i18n.en` dictionaries.

---
Thank you for helping make open-source security accessible to everyone! 🚀
