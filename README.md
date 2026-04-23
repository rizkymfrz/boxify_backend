# Boxify Backend

FastAPI backend for the Boxify annotation tool.

## Prerequisites

- Python 3.10+
- MySQL 8.0+

## Installation

```bash
cd boxify_backend
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate
pip install -r requirements.txt
```

## Environment Setup

```bash
cp .env.example .env
```

Open `.env` and update the database connection URL:

```env
MYSQL_URL=mysql+pymysql://<user>:<password>@localhost:3306/boxify
```

## Running the Server

```bash
uvicorn api.main:app --reload --port 8000
```

Database tables are created automatically on first startup. API docs available at [http://localhost:8000/docs](http://localhost:8000/docs).
