# router

## 실행하는 법법

### 1. uv 설치
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 의존성 설치
```bash
# uv로 프로젝트 환경 설정 (pyproject.toml 기준 dependency 다운됨됨)
uv sync
```

### 3. 실행
```bash
# example.py 실행
uv run python example.py
```

또는

```bash
source .venv/bin/activate
python example.py
```
