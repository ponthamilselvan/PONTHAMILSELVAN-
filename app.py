from database import db
"""PocketSmart AI - FastAPI backend (clean build)"""
import os, uuid, shutil, hmac, hashlib, base64
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import (FastAPI, HTTPException, Depends, File, UploadFile, Form,
                     Request, status, Cookie)
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
import jwt
from jwt import PyJWTError as JWTError
from dotenv import load_dotenv

# CRITICAL: load .env BEFORE importing gemini_utils
from pathlib import Path as _Path
load_dotenv(dotenv_path=_Path(__file__).resolve().parent / ".env", override=True)

from gemini_utils import (get_home_recommendations,
                          get_party_recommendations,
                          get_jewelry_recommendations)

SECRET_KEY = os.getenv("SECRET_KEY", "pocketsmart-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="PocketSmart AI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# ---------- In-memory storage ----------
users_db: Dict[str, Dict[str, Any]] = {}
active_sessions: Dict[str, Dict[str, Any]] = {}
user_recommendations: Dict[str, List[Dict[str, Any]]] = {}
blacklisted_tokens: set = set()


# ---------- Password helpers (PBKDF2, no external deps) ----------
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return base64.b64encode(salt + dk).decode()

def verify_password(password: str, stored: str) -> bool:
    try:
        raw = base64.b64decode(stored.encode())
        salt, dk = raw[:16], raw[16:]
        test = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
        return hmac.compare_digest(dk, test)
    except Exception:
        return False


# ---------- Pydantic schemas ----------
class HomeBudgetInput(BaseModel):
    total_budget: float
    room_type: Optional[str] = "Living Room"
    num_lights: int = 0
    num_fans: int = 0
    num_furniture: int = 0
    num_dining_tables: int = 0

class PartyBudgetInput(BaseModel):
    total_budget: float
    party_type: str = "Birthday"
    num_guests: int = 10
    venue_type: Optional[str] = ""
    needs_catering: bool = True
    needs_decoration: bool = True
    needs_entertainment: bool = False
    additional_requirements: Optional[str] = ""

class JewelryBudgetInput(BaseModel):
    total_budget: float
    occasion: str = "Wedding"
    preferences: Optional[str] = ""


# ---------- JWT ----------
def create_access_token(data, expires_delta=None):
    to_encode = data.copy()
    exp = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": exp})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_token(request: Request, access_token: Optional[str] = Cookie(None)):
    if access_token:
        return access_token
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1]
    return None

def authenticate_user(username, password):
    u = users_db.get(username)
    if not u:
        return None
    if not verify_password(password, u["hashed_password"]):
        return None
    return u


# ---------- Session helpers ----------
def _create_or_update_session(username, token):
    ex = active_sessions.get(username)
    data = ex["user_data"] if ex else {}
    if ex and ex.get("token"):
        blacklisted_tokens.add(ex["token"])
    active_sessions[username] = {
        "username": username,
        "login_time": datetime.utcnow(),
        "last_activity": datetime.utcnow(),
        "token": token,
        "user_data": data,
    }

def _save_to_history(username, rec_type, input_data, result):
    user_recommendations.setdefault(username, []).append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "recommendation_type": rec_type,
        "input_data": input_data,
        "full_result": result,
    })

def _current_username(request: Request, token: Optional[str] = Depends(get_token)):
    """Return username if valid cookie, else None."""
    if not token or token in blacklisted_tokens:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        u = payload.get("sub")
        if u and u in users_db:
            return u
    except JWTError:
        pass
    return None


# ==================================================================
# PAGE ROUTES
# ==================================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"user": None})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, error: Optional[str] = None, success: Optional[str] = None):
    return templates.TemplateResponse(request, "register.html",
                                      {"user": None, "error": error, "success": success})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None, success: Optional[str] = None):
    return templates.TemplateResponse(request, "login.html",
                                      {"user": None, "error": error, "success": success})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, username: Optional[str] = Depends(_current_username)):
    if not username:
        return RedirectResponse("/login", status_code=302)
    recent = list(reversed(user_recommendations.get(username, [])))[:5]
    return templates.TemplateResponse(request, "dashboard.html",
                                      {"user": users_db[username], "recent": recent})

@app.get("/home-planner", response_class=HTMLResponse)
async def hp(request: Request, username: Optional[str] = Depends(_current_username)):
    if not username:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "home_planner.html",
                                      {"user": users_db[username]})

@app.get("/party-planner", response_class=HTMLResponse)
async def pp(request: Request, username: Optional[str] = Depends(_current_username)):
    if not username:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "party_planner.html",
                                      {"user": users_db[username]})

@app.get("/jewelry-planner", response_class=HTMLResponse)
async def jp(request: Request, username: Optional[str] = Depends(_current_username)):
    if not username:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "jewelry_planner.html",
                                      {"user": users_db[username]})

@app.get("/history", response_class=HTMLResponse)
async def hp_history(request: Request, username: Optional[str] = Depends(_current_username)):
    if not username:
        return RedirectResponse("/login", status_code=302)
    items = list(reversed(user_recommendations.get(username, [])))
    return templates.TemplateResponse(request, "history.html",
                                      {"user": users_db[username], "history": items})


# ==================================================================
# AUTH ROUTES (plain HTML forms - no JS)
# ==================================================================
@app.post("/register")
async def register_user_form(
    username: str = Form(...),
    email: str = Form(...),
    full_name: Optional[str] = Form(None),
    password: str = Form(...),
):
    if username in users_db:
        return RedirectResponse("/register?error=Username+already+exists", status_code=302)
    if any(u["email"] == email for u in users_db.values()):
        return RedirectResponse("/register?error=Email+already+registered", status_code=302)
    users_db[username] = {
        "username": username,
        "email": email,
        "full_name": (full_name or username).strip(),
        "hashed_password": hash_password(password),
    }
    print(f"[PocketSmart] New user registered: {username}")
    return RedirectResponse("/login?success=Registration+successful.+Please+sign+in.", status_code=302)


@app.post("/login")
async def login_form(username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(username, password)
    if not user:
        print(f"[PocketSmart] Login failed for: {username}")
        return RedirectResponse("/login?error=Invalid+username+or+password", status_code=302)
    token = create_access_token({"sub": user["username"]},
                                 timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    _create_or_update_session(user["username"], token)
    print(f"[PocketSmart] Login OK: {username}")
    resp = RedirectResponse("/dashboard", status_code=302)
    resp.set_cookie("access_token", token, httponly=True,
                    max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60, samesite="lax")
    return resp


@app.post("/logout")
async def logout(request: Request, token: Optional[str] = Depends(get_token)):
    if token:
        blacklisted_tokens.add(token)
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            u = payload.get("sub")
            if u and u in active_sessions:
                del active_sessions[u]
        except JWTError:
            pass
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("access_token")
    return resp


@app.get("/session-info")
async def session_info(username: Optional[str] = Depends(_current_username)):
    if not username:
        raise HTTPException(401, "Not authenticated")
    s = active_sessions.get(username)
    if not s:
        raise HTTPException(404, "No active session")
    return {
        "username": s["username"],
        "login_time": s["login_time"].isoformat(),
        "last_activity": s["last_activity"].isoformat(),
        "session_duration_minutes": round(
            (datetime.utcnow() - s["login_time"]).total_seconds() / 60, 2),
        "user_data": s["user_data"],
    }


# ==================================================================
# PLANNER ROUTES
# ==================================================================
@app.post("/generate-home")
async def generate_home(inp: HomeBudgetInput, username: Optional[str] = Depends(_current_username)):
    if not username:
        raise HTTPException(401, "Not authenticated. Please log in.")
    result = get_home_recommendations(inp)
    _save_to_history(username, "home", inp.dict(), result)
    return result

@app.post("/generate-party")
async def generate_party(inp: PartyBudgetInput, username: Optional[str] = Depends(_current_username)):
    if not username:
        raise HTTPException(401, "Not authenticated. Please log in.")
    result = get_party_recommendations(inp)
    _save_to_history(username, "party", inp.dict(), result)
    return result

@app.post("/generate-jewelry")
async def generate_jewelry(
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    username: Optional[str] = Depends(_current_username),
):
    if not username:
        raise HTTPException(401, "Not authenticated. Please log in.")
    image_path = None
    if image and image.filename:
        safe = f"{uuid.uuid4().hex}_{os.path.basename(image.filename)}"
        image_path = os.path.join(UPLOAD_DIR, safe)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
    inp = JewelryBudgetInput(total_budget=total_budget, occasion=occasion,
                             preferences=preferences or "")
    result = get_jewelry_recommendations(inp, image_path)
    _save_to_history(username, "jewelry",
                     {**inp.dict(), "has_image": bool(image_path)}, result)
    return result


@app.get("/recommendations")
async def list_recs(username: Optional[str] = Depends(_current_username)):
    if not username:
        raise HTTPException(401, "Not authenticated")
    return {"recommendations": user_recommendations.get(username, [])}


# ==================================================================
# STARTUP
# ==================================================================
@app.on_event("startup")
async def on_startup():
    # Seed demo user for instant login
    if "sai" not in users_db:
        users_db["sai"] = {
            "username": "infy",
            "email": "infy@test.com",
            "full_name": "infinity",
            "hashed_password": hash_password("INFY1234"),
        }
        print("[PocketSmart] Demo user created: sai / test1234")
    print("=" * 60)
    print("  PocketSmart AI  |  FastAPI + Gemini 1.5 Flash Pro")
    print("  Open:  http://127.0.0.1:8000")
    print("  Login: infy / INFY1234")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    print("Starting PocketSmart: AI Budget Planner...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
