### Python Version ###
3.13.x

Use **Python 3.13**, not 3.14. PaddleOCR’s engine (`paddlepaddle`) has Windows wheels only up to 3.13.

---

### SET UP DATABASE ###

**Windows**

```sql
psql -U postgres -h localhost
```

Then paste:

```sql
CREATE DATABASE gif_db;
CREATE USER gif WITH PASSWORD '2+PJh#&?';
ALTER ROLE gif SET client_encoding TO 'utf8';
ALTER ROLE gif SET default_transaction_isolation TO 'read committed';
ALTER ROLE gif SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE gif_db TO gif;
\c gif_db;
ALTER SCHEMA public OWNER TO gif;
GRANT ALL ON SCHEMA public TO gif;
\q
```

---

### Local settings (API keys) ###

Django loads `gif/core/settings/base.py`, then optionally `gif/core/settings/local.py`.  
`local.py` is gitignored — put secrets there, not in `base.py`.

From the repo root:

```powershell
copy gif\core\settings\local.py.example gif\core\settings\local.py
```

Open `gif/core/settings/local.py` and fill in the keys you use:

```python
OCRSPACE_API_KEY = 'your-ocrspace-key'
REPLICATE_API_TOKEN = 'r8_...'
# OPENAI_API_KEY = ''
# GOOGLE_APPLICATION_CREDENTIALS = r'C:\path\to\credentials.json'
```

- **OCR.space** — default text detector (`DETECTION_TEXT_BACKEND = 'ocrspace'`).
- **Replicate** — default object detector (`DETECTION_OBJECT_BACKEND = 'replicate'`). Also used if local SAM is missing for card cut-outs.
- **OpenAI / Google** — only if you switch backends to `gpt4o` or `google`.

Do not commit `local.py`.

---

### Python environment ###

From the repo root (`Project/gif`):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements\base.txt
```

Confirm the interpreter is 3.13:

```powershell
python -c "import sys; print(sys.version)"
```

---

### Node (Remotion GIF render) ###

Install [Node.js](https://nodejs.org/), then:

```powershell
cd remotion
npm install
cd ..
```

---

### SAM weights (optional, local cut-outs) ###

Person / card silhouettes use SAM 2.1 if the weights exist:

- `gif/ml_models/sam2.1_t.pt`, or
- `gif/sam2.1_t.pt`

If the file is missing, card cut-out falls back to Replicate rembg (needs `REPLICATE_API_TOKEN`).

---

### Migrate and run ###

Always run Django from the inner `gif` folder (where `manage.py` lives), with the venv active:

```powershell
cd gif
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/

In Cursor, set the interpreter to `.venv\Scripts\python.exe` so Run/Debug uses 3.13 as well.
