from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from starlette.middleware.sessions import SessionMiddleware

import pandas as pd
import io
import csv
import os

from database import SessionLocal, engine, Base
from models import Vehicle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---------- Session middleware for login ----------
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET_KEY", "change-this-session-secret"),
)

# ---------- In-memory cache for pricing ----------
last_pricing_results: list[dict] = []
last_pricing_input_rows: list[dict] = []


# ---------- Helpers ----------

def clean_int(value):
    """Convert to int or return None if blank/NaN."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(float(str(value).replace(",", "").strip()))
    except Exception:
        return None


def clean_float(value):
    """Convert to float or return None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except Exception:
        return None


def parse_title_for_year_make_model_and_status(title: str):
    """
    From a Title like 'Title - 2019 HYUNDAI ACCENT', extract:
    - year
    - make
    - model
    - title_status ('Title', 'Parts Only', or None)
    """
    if not title:
        return None, None, None, None

    t_lower = title.lower()
    status = None
    if "parts" in t_lower:
        status = "Parts Only"
    elif "title" in t_lower:
        status = "Title"

    # Strip leading "Title - " or similar
    cleaned = title
    if "-" in cleaned:
        parts = cleaned.split("-", 1)
        cleaned = parts[1].strip()
    tokens = cleaned.split()

    year = None
    make = None
    model = None

    for i, tok in enumerate(tokens):
        if tok.isdigit() and len(tok) == 4:
            year = clean_int(tok)
            if i + 1 < len(tokens):
                make = tokens[i + 1]
            if i + 2 < len(tokens):
                model = " ".join(tokens[i + 2:])
            break

    return year, make, model, status


def build_stats(
    db: Session,
    year: int | None,
    make: str | None,
    model: str | None,
    state: str | None,
    company: str | None,
    min_mileage_val: int | None,
    max_mileage_val: int | None,
):
    """Shared logic for computing stats and returning matching vehicles."""
    stats = None
    vehicles = []

    if (
        year
        or make
        or model
        or state
        or company
        or min_mileage_val is not None
        or max_mileage_val is not None
    ):
        query = db.query(Vehicle)

        if year:
            query = query.filter(Vehicle.year == year)
        if make:
            query = query.filter(Vehicle.make.ilike(make.strip()))
        if model:
            query = query.filter(Vehicle.model.ilike(model.strip()))
        if state:
            query = query.filter(Vehicle.state == state)
        if company:
            query = query.filter(Vehicle.company == company)
        if min_mileage_val is not None:
            query = query.filter(Vehicle.mileage >= min_mileage_val)
        if max_mileage_val is not None:
            query = query.filter(Vehicle.mileage <= max_mileage_val)

        vehicles = query.all()
        if vehicles:
            prices = [v.sale_price for v in vehicles if v.sale_price is not None]
            if prices:
                stats = {
                    "count": len(prices),
                    "avg_price": sum(prices) / len(prices),
                    "high_price": max(prices),
                    "low_price": min(prices),
                }

    return stats, vehicles


def is_logged_in(request: Request) -> bool:
    """Simple session flag check."""
    return bool(request.session.get("logged_in"))


# ---------- DB setup on startup ----------

@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)


# ---------- DB session dependency ----------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ======================================================================
#                            AUTH ROUTES
# ======================================================================

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": None,
        },
    )


@app.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    password: str = Form(...),
):
    # Password from environment variable; default is "changeme" for local dev
    expected = os.environ.get("APP_PASSWORD", "changeme")
    if password == expected:
        request.session["logged_in"] = True
        return RedirectResponse(url="/", status_code=302)

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": "Invalid password.",
        },
    )


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


# ======================================================================
#                            ROUTES
# ======================================================================

# ---------- Home ----------

@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)

    total_vehicles = db.query(Vehicle).count()
    distinct_years = db.query(func.count(func.distinct(Vehicle.year))).scalar()
    distinct_makes = db.query(func.count(func.distinct(Vehicle.make))).scalar()
    distinct_models = db.query(func.count(func.distinct(Vehicle.model))).scalar()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "vehicle_count": total_vehicles,
            "year_count": distinct_years,
            "make_count": distinct_makes,
            "model_count": distinct_models,
        },
    )


# ---------- Upload historical data ----------

@app.api_route("/upload-history", methods=["GET", "POST"], response_class=HTMLResponse)
async def upload_history(
    request: Request,
    file: UploadFile | None = File(None),
    state: str | None = Form(None),
    company: str | None = Form(None),
    db: Session = Depends(get_db),
):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)

    # GET: just show the form
    if request.method == "GET":
        return templates.TemplateResponse(
            "upload_history.html",
            {
                "request": request,
                "message": None,
                "error": None,
            },
        )

    # POST: process upload
    if not file or not file.filename:
        return templates.TemplateResponse(
            "upload_history.html",
            {
                "request": request,
                "message": None,
                "error": "No file uploaded.",
            },
        )

    content = await file.read()

    # Try CSV first, then Excel if needed
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        try:
            df = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            return templates.TemplateResponse(
                "upload_history.html",
                {
                    "request": request,
                    "message": None,
                    "error": f"Could not read file as CSV or Excel: {e}",
                },
            )

    imported = 0
    errors = 0
    first_error = None

    for _, row in df.iterrows():
        try:
            vin_raw = row.get("VIN") or row.get("Vin") or row.get("vin")
            vin = str(vin_raw).strip() if vin_raw is not None else ""

            # Skip rows with missing or bogus VIN (NaN, blank, etc.)
            if not vin or vin.lower() == "nan":
                errors += 1
                if first_error is None:
                    first_error = "Missing VIN"
                continue

            # Skip if this VIN already exists (unique index)
            existing = db.query(Vehicle).filter(Vehicle.vin == vin).first()
            if existing:
                errors += 1
                if first_error is None:
                    first_error = f"Duplicate VIN skipped: {vin}"
                continue

            # Try to get year/make/model directly or from "Title"
            title_val = row.get("Title") or row.get("title") or ""
            year = clean_int(row.get("Year") or row.get("year"))
            make = (row.get("Make") or row.get("make") or "").strip() or None
            model = (row.get("Model") or row.get("model") or "").strip() or None
            title_status = None

            if title_val:
                py, pmake, pmodel, pstatus = parse_title_for_year_make_model_and_status(
                    str(title_val)
                )
                # Only fill in if missing
                if year is None and py is not None:
                    year = py
                if not make and pmake:
                    make = pmake
                if not model and pmodel:
                    model = pmodel
                if pstatus:
                    title_status = pstatus

            # If still nothing for title_status, maybe Title has 'Parts Only' etc.
            if not title_status and title_val:
                t_lower = str(title_val).lower()
                if "parts" in t_lower:
                    title_status = "Parts Only"
                elif "title" in t_lower:
                    title_status = "Title"

            # Enforce YEAR present (DB has NOT NULL constraint)
            if year is None:
                errors += 1
                if first_error is None:
                    first_error = f"Missing or invalid year for VIN {vin}"
                continue

            sale_price = clean_float(
                row.get("Price")
                or row.get("Sale Price")
                or row.get("SalePrice")
                or row.get("price")
            )

            mileage = clean_int(
                row.get("Mileage") or row.get("mileage") or row.get("Odometer")
            )

            # Key/run/drive as text
            has_keys = row.get("Keys") or row.get("Key") or row.get("keys")
            if hasattr(pd, "isna") and pd.isna(has_keys):
                has_keys = None
            if has_keys is not None:
                has_keys = str(has_keys).strip()

            runs = row.get("Run") or row.get("Runs") or row.get("run")
            if hasattr(pd, "isna") and pd.isna(runs):
                runs = None
            if runs is not None:
                runs = str(runs).strip()

            drives = row.get("Drive") or row.get("Drives") or row.get("drive")
            if hasattr(pd, "isna") and pd.isna(drives):
                drives = None
            if drives is not None:
                drives = str(drives).strip()

            # State / company from columns if present, otherwise from form
            row_state = (
                row.get("State")
                or row.get("state")
                or row.get("Location State")
                or row.get("Pickup State")
            )
            if hasattr(pd, "isna") and pd.isna(row_state):
                row_state = None
            if row_state is not None:
                row_state = str(row_state).strip()
            if not row_state and state:
                row_state = state.strip()

            row_company = (
                row.get("Company")
                or row.get("company")
                or row.get("Seller Name")
                or row.get("Seller Company")
            )
            if hasattr(pd, "isna") and pd.isna(row_company):
                row_company = None
            if row_company is not None:
                row_company = str(row_company).strip()
            if not row_company and company:
                row_company = company.strip()

            vehicle = Vehicle(
                vin=vin,
                year=year,
                make=make,
                model=model,
                mileage=mileage,
                has_keys=has_keys,
                runs=runs,
                drives=drives,
                sale_price=sale_price,
                sale_date=None,
                title_status=title_status,
                source_file=file.filename,
                state=row_state,
                company=row_company,
            )
            db.add(vehicle)
            imported += 1

        except Exception as e:
            errors += 1
            if first_error is None:
                first_error = f"Row error: {e}"

    # Commit with safety net so we DON'T throw raw 500s
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        return templates.TemplateResponse(
            "upload_history.html",
            {
                "request": request,
                "message": None,
                "error": f"Database constraint error while saving rows: {e}",
            },
        )
    except Exception as e:
        db.rollback()
        return templates.TemplateResponse(
            "upload_history.html",
            {
                "request": request,
                "message": None,
                "error": f"Unexpected database error: {e}",
            },
        )

    msg = f"Imported {imported} rows from {file.filename}. {errors} row(s) had errors and were skipped."
    if first_error:
        msg += f" First error: {first_error}"

    return templates.TemplateResponse(
        "upload_history.html",
        {
            "request": request,
            "message": msg,
            "error": None,
        },
    )


# ---------- View history ----------

@app.get("/history", response_class=HTMLResponse)
def view_history(
    request: Request,
    state: str | None = None,
    company: str | None = None,
    db: Session = Depends(get_db),
):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)

    query = db.query(Vehicle)

    if state:
        query = query.filter(Vehicle.state == state)
    if company:
        query = query.filter(Vehicle.company == company)

    vehicles = query.order_by(Vehicle.id.desc()).limit(1000).all()

    # For filter dropdowns, get distinct states/companies
    states = [s[0] for s in db.query(Vehicle.state).distinct().all() if s[0]]
    companies = [c[0] for c in db.query(Vehicle.company).distinct().all() if c[0]]

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "vehicles": vehicles,
            "states": states,
            "companies": companies,
            "selected_state": state,
            "selected_company": company,
        },
    )


# ---------- Historical stats ----------

@app.get("/stats", response_class=HTMLResponse)
def stats_page(
    request: Request,
    year: int | None = None,
    make: str | None = None,
    model: str | None = None,
    state: str | None = None,
    company: str | None = None,
    min_mileage: int | None = None,
    max_mileage: int | None = None,
    db: Session = Depends(get_db),
):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)

    stats, vehicles = build_stats(
        db=db,
        year=year,
        make=make,
        model=model,
        state=state,
        company=company,
        min_mileage_val=min_mileage,
        max_mileage_val=max_mileage,
    )

    # Distinct states/companies for filters
    states = [s[0] for s in db.query(Vehicle.state).distinct().all() if s[0]]
    companies = [c[0] for c in db.query(Vehicle.company).distinct().all() if c[0]]

    return templates.TemplateResponse(
        "stats.html",
        {
            "request": request,
            "stats": stats,
            "vehicles": vehicles,
            "states": states,
            "companies": companies,
            "year": year,
            "make": make,
            "model": model,
            "selected_state": state,
            "selected_company": company,
            "min_mileage": min_mileage,
            "max_mileage": max_mileage,
        },
    )


@app.post("/stats", response_class=HTMLResponse)
async def stats_page_post(
    request: Request,
    year: str | None = Form(None),
    make: str | None = Form(None),
    model: str | None = Form(None),
    state: str | None = Form(None),
    company: str | None = Form(None),
    min_mileage: str | None = Form(None),
    max_mileage: str | None = Form(None),
    db: Session = Depends(get_db),
):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)

    year_val = clean_int(year) if year not in (None, "") else None
    min_mileage_val = clean_int(min_mileage) if min_mileage not in (None, "") else None
    max_mileage_val = clean_int(max_mileage) if max_mileage not in (None, "") else None

    make_val = make.strip() if make else None
    model_val = model.strip() if model else None
    state_val = state.strip() if state else None
    company_val = company.strip() if company else None

    stats, vehicles = build_stats(
        db=db,
        year=year_val,
        make=make_val,
        model=model_val,
        state=state_val,
        company=company_val,
        min_mileage_val=min_mileage_val,
        max_mileage_val=max_mileage_val,
    )

    states = [s[0] for s in db.query(Vehicle.state).distinct().all() if s[0]]
    companies = [c[0] for c in db.query(Vehicle.company).distinct().all() if c[0]]

    return templates.TemplateResponse(
        "stats.html",
        {
            "request": request,
            "stats": stats,
            "vehicles": vehicles,
            "states": states,
            "companies": companies,
            "year": year_val,
            "make": make_val,
            "model": model_val,
            "selected_state": state_val,
            "selected_company": company_val,
            "min_mileage": min_mileage_val,
            "max_mileage": max_mileage_val,
        },
    )


# ---------- Price vehicles for purchase ----------

@app.get("/price-vehicles", response_class=HTMLResponse)
def price_vehicles_form(request: Request):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)

    return templates.TemplateResponse(
        "price_vehicles.html",
        {
            "request": request,
            "results": None,
            "margin": 20.0,
            "expenses": 0.0,
            "state": None,
            "company": None,
            "min_mileage": None,
            "max_mileage": None,
            "error": None,
        },
    )


@app.post("/price-vehicles", response_class=HTMLResponse)
async def price_vehicles(
    request: Request,
    file: UploadFile | None = File(None),
    margin: float = Form(20.0),
    expenses: float = Form(0.0),
    # Optional filters for comps (strings are fine)
    state: str | None = Form(None),
    company: str | None = Form(None),
    min_mileage: str | None = Form(None),
    max_mileage: str | None = Form(None),
    # Optional manual single vehicle:
    manual_year: int | None = Form(None),
    manual_make: str | None = Form(None),
    manual_model: str | None = Form(None),
    manual_mileage: str | None = Form(None),
    db: Session = Depends(get_db),
):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)

    """
    Price vehicles based on historical comps in the Vehicle table.
    - Can take a file of vehicles OR a single manually-entered vehicle.
    - If neither is provided, re-uses the last uploaded/entered vehicles.
    """
    global last_pricing_results, last_pricing_input_rows

    # Normalize text filters (strip whitespace, treat empty as None)
    state = state.strip() or None if state else None
    company = company.strip() or None if company else None

    # Convert mileage inputs from strings to ints (or None if blank)
    min_mileage_val = clean_int(min_mileage) if min_mileage not in (None, "") else None
    max_mileage_val = clean_int(max_mileage) if max_mileage not in (None, "") else None
    manual_mileage_val = (
        clean_int(manual_mileage) if manual_mileage not in (None, "") else None
    )

    purchase_rows: list[dict] = []

    # 1) If a new file is uploaded, parse it and REPLACE last input rows
    if file and file.filename:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            try:
                df = pd.read_excel(io.BytesIO(content))
            except Exception as e:
                return templates.TemplateResponse(
                    "price_vehicles.html",
                    {
                        "request": request,
                        "results": None,
                        "margin": margin,
                        "expenses": expenses,
                        "state": state,
                        "company": company,
                        "min_mileage": min_mileage,
                        "max_mileage": max_mileage,
                        "error": f"Could not read file as CSV or Excel: {e}",
                    },
                )

        for _, row in df.iterrows():
            vin = str(
                row.get("VIN") or row.get("Vin") or row.get("vin") or ""
            ).strip()
            year = clean_int(row.get("Year") or row.get("year"))
            make = (row.get("Make") or row.get("make") or "").strip() or None
            model = (row.get("Model") or row.get("model") or "").strip() or None
            mileage = clean_int(
                row.get("Mileage") or row.get("mileage") or row.get("Odometer")
            )

            # Require year/make/model to identify a vehicle
            if not (year and make and model):
                continue

            purchase_rows.append(
                {
                    "vin": vin,
                    "year": year,
                    "make": make,
                    "model": model,
                    "mileage": mileage,
                }
            )

        last_pricing_input_rows = purchase_rows

    # 2) No file, but manual single vehicle was entered
    elif manual_year and manual_make and manual_model:
        purchase_rows.append(
            {
                "vin": "",
                "year": manual_year,
                "make": manual_make.strip(),
                "model": manual_model.strip(),
                "mileage": manual_mileage_val,
            }
        )
        last_pricing_input_rows = purchase_rows

    # 3) Neither file nor manual input: re-use last input rows (recalc)
    else:
        purchase_rows = last_pricing_input_rows or []

    # If after all of that we still have nothing to price, show an error
    if not purchase_rows:
        return templates.TemplateResponse(
            "price_vehicles.html",
            {
                "request": request,
                "results": None,
                "margin": margin,
                "expenses": expenses,
                "state": state,
                "company": company,
                "min_mileage": min_mileage,
                "max_mileage": max_mileage,
                "error": "No vehicles to price. Upload a file or enter a vehicle manually.",
            },
        )

    # Build pricing results from historical comps
    results: list[dict] = []

    for row in purchase_rows:
        year = row["year"]
        make = row["make"]
        model = row["model"]
        mileage = row["mileage"]

        # Base query: year + make + model
        query = db.query(Vehicle).filter(
            Vehicle.year == year,
            Vehicle.make.ilike(f"%{make}%"),
            Vehicle.model.ilike(f"%{model}%"),
        )

        # Optional filters on comps
        if state:
            query = query.filter(Vehicle.state == state)
        if company:
            query = query.filter(Vehicle.company == company)
        if min_mileage_val is not None:
            query = query.filter(Vehicle.mileage >= min_mileage_val)
        if max_mileage_val is not None:
            query = query.filter(Vehicle.mileage <= max_mileage_val)

        comps = query.all()

        avg_price = None
        high_price = None
        low_price = None
        target_price = None
        max_offer = None

        prices = [v.sale_price for v in comps if v.sale_price is not None]
        if prices:
            avg_price = sum(prices) / len(prices)
            high_price = max(prices)
            low_price = min(prices)
            # simple logic: subtract expected expenses, then apply margin
            target_price = avg_price - expenses
            if target_price < 0:
                target_price = 0
            max_offer = target_price * (1 - margin / 100.0)

        result = {
            "vin": row["vin"],
            "year": year,
            "make": make,
            "model": model,
            "mileage": mileage,
            "avg_price": avg_price,
            "high_price": high_price,
            "low_price": low_price,
            "target_price": target_price,
            "max_offer": max_offer,
        }
        results.append(result)

    last_pricing_results = results

    return templates.TemplateResponse(
        "price_vehicles.html",
        {
            "request": request,
            "results": results,
            "margin": margin,
            "expenses": expenses,
            "state": state,
            "company": company,
            "min_mileage": min_mileage,
            "max_mileage": max_mileage,
            "error": None,
        },
    )


# ---------- Export pricing results as CSV ----------

@app.get("/export-pricing-csv")
def export_pricing_csv(request: Request):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)

    if not last_pricing_results:
        return Response(
            "No pricing results to export.", media_type="text/plain"
        )

    output = io.StringIO()
    writer = csv.writer(output)

    # Header row (includes mileage)
    writer.writerow(
        [
            "VIN",
            "Year",
            "Make",
            "Model",
            "Mileage",
            "Average Sale Price",
            "High Sale Price",
            "Low Sale Price",
            "Target Price",
            "Max Offer",
        ]
    )

    for row in last_pricing_results:
        writer.writerow(
            [
                row.get("vin", ""),
                row.get("year", ""),
                row.get("make", ""),
                row.get("model", ""),
                row.get("mileage", ""),
                row.get("avg_price", ""),
                row.get("high_price", ""),
                row.get("low_price", ""),
                row.get("target_price", ""),
                row.get("max_offer", ""),
            ]
        )

    csv_data = output.getvalue()
    output.close()

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=pricing_results.csv"
        },
    )

