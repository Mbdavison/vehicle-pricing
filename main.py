from typing import List, Optional

from fastapi import FastAPI, Depends, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func, not_
from sqlalchemy.exc import IntegrityError
import pandas as pd
import io
import os
import csv

app = FastAPI()

@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)

from database import SessionLocal, engine, Base
import models

# Create DB tables if they don't exist


# Jinja2 templates directory
templates = Jinja2Templates(directory="templates")


# DB session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>Vehicle Pricing Tool</title></head>
        <body>
            <h1>Vehicle Pricing Tool</h1>
            <ul>
                <li><a href="/test-db">Test database connection</a></li>
                <li><a href="/upload-history">Upload vehicle sales history</a></li>
                <li><a href="/history">Browse upload history</a></li>
                <li><a href="/stats">View historical stats</a></li>
                <li><a href="/price-vehicles">Price vehicles for purchase</a></li>
            </ul>
        </body>
    </html>
    """


@app.get("/test-db")
def test_db(db: Session = Depends(get_db)):
    count = db.query(models.VehicleSold).count()
    return {"vehicle_count": count}


@app.get("/upload-history", response_class=HTMLResponse)
def upload_history_form(request: Request):
    return templates.TemplateResponse(
        "upload_history.html",
        {"request": request, "message": None}
    )


@app.post("/upload-history", response_class=HTMLResponse)
async def upload_history(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Read uploaded file content
    content = await file.read()
    ext = os.path.splitext(file.filename)[1].lower()

    # Try Excel if extension looks like Excel, else CSV
    try:
        if ext in [".xls", ".xlsx"]:
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return templates.TemplateResponse(
            "upload_history.html",
            {"request": request, "message": f"Error reading file: {e}"}
        )

    # Normalize column names
    df.columns = df.columns.str.lower()
    cols = set(df.columns)

    # If we have a "title" column like "Title - 2019 HYUNDAI ACCENT",
    # derive year/make/model and title_status if possible.
    if "title" in cols:

        def parse_title_and_status(title: str):
            """
            Example inputs:
              "Title - 2019 HYUNDAI ACCENT"
              "Parts Only - 2014 FORD FOCUS"
            Returns: (year, make, model, status)
            """
            if pd.isna(title):
                return None, None, None, None

            text = str(title)
            status = None

            # Split on first "-"
            if "-" in text:
                prefix, rest = text.split("-", 1)
                status = prefix.strip() or None
                text = rest
            text = text.strip()

            parts = text.split()
            # Expect: YEAR MAKE MODEL...
            if len(parts) < 2:
                return None, None, None, status

            try:
                year = int(parts[0])
            except Exception:
                return None, None, None, status

            make = parts[1]
            model = " ".join(parts[2:]) if len(parts) > 2 else ""

            return year, make, model, status

        parsed = df["title"].apply(parse_title_and_status)

        # Only fill in year/make/model where they don't already exist
        if "year" not in cols:
            df["year"] = parsed.apply(lambda x: x[0])
        if "make" not in cols:
            df["make"] = parsed.apply(lambda x: x[1])
        if "model" not in cols:
            df["model"] = parsed.apply(lambda x: x[2])

        df["title_status_from_title"] = parsed.apply(lambda x: x[3])

    # If there's a "price" column but no "sale_price", derive it (strip $ and ,)
    if "price" in cols and "sale_price" not in cols:
        df["sale_price"] = (
            df["price"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .replace("", pd.NA)
        )
        df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")

    # Required columns (AFTER all derivation)
    required = {"vin", "year", "make", "model", "sale_price"}
    cols = set(df.columns)
    missing = required - cols

    if missing:
        return templates.TemplateResponse(
            "upload_history.html",
            {
                "request": request,
                "message": (
                    "Missing required columns after processing: "
                    f"{', '.join(sorted(missing))}."
                ),
            }
        )

    inserted = 0
    skipped = 0
    errors = 0

    for _, row in df.iterrows():
        try:
            # Skip rows missing critical fields
            if pd.isna(row.get("vin")) or pd.isna(row.get("year")) or pd.isna(row.get("sale_price")):
                errors += 1
                continue

            vin = str(row["vin"]).strip().upper()
            year = int(row["year"])
            make = str(row["make"]).strip()
            model = str(row["model"]).strip()
            sale_price = float(row["sale_price"])

            # Mileage (integer if present)
            mileage = None
            if "mileage" in df.columns and not pd.isna(row.get("mileage", None)):
                try:
                    mileage = int(float(row["mileage"]))
                except ValueError:
                    mileage = None

            # Store raw text for key / runs / drives
            has_keys = None
            if "key" in df.columns and not pd.isna(row.get("key", None)):
                has_keys = str(row["key"]).strip()

            runs = None
            if "runs" in df.columns and not pd.isna(row.get("runs", None)):
                runs = str(row["runs"]).strip()

            drives = None
            if "drives" in df.columns and not pd.isna(row.get("drives", None)):
                drives = str(row["drives"]).strip()

            # sale_date if present
            sale_date = None
            if "sale_date" in df.columns and not pd.isna(row.get("sale_date", None)):
                try:
                    sale_date = pd.to_datetime(row["sale_date"]).date()
                except Exception:
                    sale_date = None

            # title_status:
            # 1) explicit title_status column if present
            # 2) otherwise, title_status_from_title (e.g., "Title", "Parts Only")
            title_status = None
            if "title_status" in df.columns and not pd.isna(row.get("title_status", None)):
                title_status = str(row["title_status"]).strip()
            elif "title_status_from_title" in df.columns and not pd.isna(
                row.get("title_status_from_title", None)
            ):
                title_status = str(row["title_status_from_title"]).strip()

            # State (e.g. seller state) if present
            state = None
            if "state" in df.columns and not pd.isna(row.get("state", None)):
                state = str(row["state"]).strip()

            # Company (e.g. seller name or company) if present
            company = None
            if "seller name" in df.columns and not pd.isna(row.get("seller name", None)):
                company = str(row["seller name"]).strip()
            elif "company" in df.columns and not pd.isna(row.get("company", None)):
                company = str(row["company"]).strip()

            vehicle = models.VehicleSold(
                vin=vin,
                year=year,
                make=make,
                model=model,
                sale_price=sale_price,
                mileage=mileage,
                has_keys=has_keys,
                runs=runs,
                drives=drives,
                sale_date=sale_date,
                title_status=title_status,
                source_file=file.filename,
                state=state,
                company=company,
            )

            db.add(vehicle)

            try:
                db.commit()
                inserted += 1
            except IntegrityError:
                db.rollback()
                skipped += 1
            except Exception:
                db.rollback()
                errors += 1

        except Exception:
            errors += 1

    message = f"Imported {inserted} rows from {file.filename}."
    if skipped:
        message += f" Skipped {skipped} duplicate VIN(s)."
    if errors:
        message += f" {errors} row(s) had missing or invalid data and were skipped."

    return templates.TemplateResponse(
        "upload_history.html",
        {"request": request, "message": message}
    )


# ---------- HISTORY BROWSER ----------

@app.get("/history", response_class=HTMLResponse)
def history_form(request: Request):
    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "rows": None,
            "message": None,
            "year": "",
            "make": "",
            "model": "",
            "state": "",
            "company": "",
            "sort_by": "sale_date",
            "sort_dir": "desc",
        },
    )


@app.post("/history", response_class=HTMLResponse)
def history_results(
    request: Request,
    year: Optional[str] = Form(None),
    make: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    state: Optional[str] = Form(None),
    company: Optional[str] = Form(None),
    sort_by: str = Form("sale_date"),
    sort_dir: str = Form("desc"),
    db: Session = Depends(get_db),
):
    year_str = (year or "").strip()
    make = (make or "").strip()
    model = (model or "").strip()
    state = (state or "").strip()
    company = (company or "").strip()

    filters = []
    message = None

    if year_str:
        try:
            year_val = int(year_str)
            filters.append(models.VehicleSold.year == year_val)
        except ValueError:
            message = "Invalid year filter; ignoring."
    if make:
        filters.append(models.VehicleSold.make.ilike(make))
    if model:
        filters.append(models.VehicleSold.model.ilike(model))
    if state:
        filters.append(models.VehicleSold.state.ilike(state))
    if company:
        filters.append(models.VehicleSold.company.ilike(company))

    query = db.query(models.VehicleSold)
    if filters:
        query = query.filter(*filters)

    # Sorting
    sort_by = sort_by or "sale_date"
    sort_dir = sort_dir or "desc"

    if sort_by == "state":
        col = models.VehicleSold.state
    elif sort_by == "company":
        col = models.VehicleSold.company
    elif sort_by == "sale_price":
        col = models.VehicleSold.sale_price
    elif sort_by == "year":
        col = models.VehicleSold.year
    else:
        col = models.VehicleSold.sale_date

    if sort_dir == "asc":
        query = query.order_by(col.asc())
    else:
        query = query.order_by(col.desc())

    rows = query.limit(200).all()

    if not rows and not message:
        message = "No records found for the given filters."

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "rows": rows,
            "message": message,
            "year": year_str,
            "make": make,
            "model": model,
            "state": state,
            "company": company,
            "sort_by": sort_by,
            "sort_dir": sort_dir,
        },
    )


# ---------- STATS ----------

@app.get("/stats", response_class=HTMLResponse)
def stats_form(request: Request):
    return templates.TemplateResponse(
        "stats.html",
        {"request": request, "stats": None, "message": None}
    )


@app.post("/stats", response_class=HTMLResponse)
def stats_results(
    request: Request,
    year: int = Form(...),
    make: str = Form(...),
    model: str = Form(...),
    state: Optional[str] = Form(None),
    company: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    make = make.strip()
    model = model.strip()
    state = (state or "").strip()
    company = (company or "").strip()

    # Build filters list dynamically so state/company are optional
    filters = [
        models.VehicleSold.year == year,
        models.VehicleSold.make.ilike(make),
        models.VehicleSold.model.ilike(model),
    ]
    if state:
        filters.append(models.VehicleSold.state.ilike(state))
    if company:
        filters.append(models.VehicleSold.company.ilike(company))

    result = db.query(
        func.avg(models.VehicleSold.sale_price).label("avg_price"),
        func.min(models.VehicleSold.sale_price).label("min_price"),
        func.max(models.VehicleSold.sale_price).label("max_price"),
        func.count(models.VehicleSold.id).label("count"),
    ).filter(*filters).one()

    if result.count == 0:
        return templates.TemplateResponse(
            "stats.html",
            {
                "request": request,
                "stats": None,
                "message": "No matching history found.",
            }
        )

    stats = {
        "year": year,
        "make": make,
        "model": model,
        "avg_price": float(result.avg_price),
        "min_price": float(result.min_price),
        "max_price": float(result.max_price),
        "count": result.count,
    }

    return templates.TemplateResponse(
        "stats.html",
        {"request": request, "stats": stats, "message": None}
    )


# ---------- PRICING WITH COMP FILTERS + MILEAGE RANGE + CSV EXPORT ----------

@app.get("/price-vehicles", response_class=HTMLResponse)
def price_form(request: Request):
    # Initial defaults
    return templates.TemplateResponse(
        "price_vehicles.html",
        {
            "request": request,
            "results": None,
            "message": None,
            "margin": 20,
            "expenses": 0,
            "comp_state": "",
            "comp_company": "",
            "comp_runs_drives_only": False,
            "comp_require_keys": False,
            "comp_exclude_parts_only": False,
            "mileage_range": 30000,
            "use_mileage_range": False,
        }
    )


@app.post("/price-vehicles", response_class=HTMLResponse)
async def price_vehicles(
    request: Request,
    margin: float = Form(...),
    expenses: float = Form(...),
    file: Optional[UploadFile] = File(None),
    # new single-vehicle fields (strings so blanks don't error)
    single_year: Optional[str] = Form(None),
    single_make: Optional[str] = Form(None),
    single_model: Optional[str] = Form(None),
    single_mileage: Optional[str] = Form(None),
    # optional filters for comps
    comp_state: Optional[str] = Form(None),
    comp_company: Optional[str] = Form(None),
    comp_runs_drives_only: Optional[str] = Form(None),
    comp_require_keys: Optional[str] = Form(None),
    comp_exclude_parts_only: Optional[str] = Form(None),
    mileage_range: Optional[float] = Form(30000),
    use_mileage_range: Optional[str] = Form(None),
    # which button: calculate vs export
    action: Optional[str] = Form("calculate"),
    # hidden fields for recalculation
    vin_list: Optional[List[str]] = Form(None),
    year_list: Optional[List[str]] = Form(None),
    make_list: Optional[List[str]] = Form(None),
    model_list: Optional[List[str]] = Form(None),
    avg_list: Optional[List[str]] = Form(None),
    comp_count_list: Optional[List[str]] = Form(None),
    db: Session = Depends(get_db),
):
    results = []

    # Normalize inputs
    single_year = (single_year or "").strip()
    single_make = (single_make or "").strip()
    single_model = (single_model or "").strip()
    single_mileage = (single_mileage or "").strip()
    comp_state = (comp_state or "").strip()
    comp_company = (comp_company or "").strip()
    action = (action or "calculate").strip().lower()

    # Checkboxes: present -> True, missing -> False
    comp_runs_drives_only = bool(comp_runs_drives_only)
    comp_require_keys = bool(comp_require_keys)
    comp_exclude_parts_only = bool(comp_exclude_parts_only)
    use_mileage_range = bool(use_mileage_range)

    try:
        mileage_range = float(mileage_range or 0)
        if mileage_range < 0:
            mileage_range = abs(mileage_range)
    except Exception:
        mileage_range = 0

    # Helper to build filters for comps based on all options
    def build_comp_filters(year: int, make: str, model: str, target_mileage: Optional[int]):
        f = [
            models.VehicleSold.year == year,
            models.VehicleSold.make.ilike(make),
            models.VehicleSold.model.ilike(model),
        ]
        if comp_state:
            f.append(models.VehicleSold.state.ilike(comp_state))
        if comp_company:
            f.append(models.VehicleSold.company.ilike(comp_company))
        if comp_runs_drives_only:
            # looks for "YES" in runs/drives text
            f.append(models.VehicleSold.runs.ilike("%yes%"))
            f.append(models.VehicleSold.drives.ilike("%yes%"))
        if comp_require_keys:
            f.append(models.VehicleSold.has_keys.ilike("%yes%"))
        if comp_exclude_parts_only:
            f.append(not_(models.VehicleSold.title_status.ilike("%parts only%")))
        if use_mileage_range and target_mileage is not None and mileage_range > 0:
            lower = max(0, int(target_mileage - mileage_range))
            upper = int(target_mileage + mileage_range)
            f.append(models.VehicleSold.mileage >= lower)
            f.append(models.VehicleSold.mileage <= upper)
        return f

    # CASE 1: New CSV/XLS uploaded -> recompute from DB (batch mode)
    if file is not None and file.filename:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            df = pd.read_excel(io.BytesIO(content))

        df.columns = df.columns.str.lower()

        required = {"vin", "year", "make", "model"}
        if required - set(df.columns):
            context = {
                "request": request,
                "results": None,
                "message": "Missing VIN, year, make, or model columns.",
                "margin": margin,
                "expenses": expenses,
                "comp_state": comp_state,
                "comp_company": comp_company,
                "comp_runs_drives_only": comp_runs_drives_only,
                "comp_require_keys": comp_require_keys,
                "comp_exclude_parts_only": comp_exclude_parts_only,
                "mileage_range": mileage_range,
                "use_mileage_range": use_mileage_range,
            }
            return templates.TemplateResponse("price_vehicles.html", context)

        for _, row in df.iterrows():
            vin = str(row["vin"]).strip().upper()
            year = int(row["year"])
            make = str(row["make"]).strip()
            model = str(row["model"]).strip()

            # target mileage for this subject vehicle (if present in upload)
            target_mileage = None
            if "mileage" in df.columns and not pd.isna(row.get("mileage", None)):
                try:
                    target_mileage = int(float(row["mileage"]))
                except Exception:
                    target_mileage = None

            filters = build_comp_filters(year, make, model, target_mileage)

            stat = db.query(
                func.avg(models.VehicleSold.sale_price).label("avg"),
                func.count(models.VehicleSold.id).label("cnt"),
            ).filter(*filters).one()

            if stat.cnt == 0 or stat.avg is None:
                results.append({
                    "vin": vin,
                    "year": year,
                    "make": make,
                    "model": model,
                    "avg_sale_price": "No comps",
                    "recommended_price": "N/A",
                    "comp_count": 0,
                })
                continue

            avg = float(stat.avg)
            rec = avg - expenses - (avg * (margin / 100))

            results.append({
                "vin": vin,
                "year": year,
                "make": make,
                "model": model,
                "avg_sale_price": round(avg, 2),
                "recommended_price": round(rec, 2),
                "comp_count": stat.cnt,
            })

    # CASE 2: Recalculate existing results (no file, no new single Y/M/M)
    elif vin_list and year_list and make_list and model_list and avg_list and comp_count_list and not (single_year or single_make or single_model):
        for vin, year, make, model, avg_str, comp_str in zip(
            vin_list, year_list, make_list, model_list, avg_list, comp_count_list
        ):
            if avg_str == "No comps":
                results.append({
                    "vin": vin,
                    "year": int(year),
                    "make": make,
                    "model": model,
                    "avg_sale_price": "No comps",
                    "recommended_price": "N/A",
                    "comp_count": int(comp_str),
                })
                continue

            avg = float(avg_str)
            rec = avg - expenses - (avg * (margin / 100))

            results.append({
                "vin": vin,
                "year": int(year),
                "make": make,
                "model": model,
                "avg_sale_price": round(avg, 2),
                "recommended_price": round(rec, 2),
                "comp_count": int(comp_str),
            })

    # CASE 3: Single year/make/model entry (no file, manual input)
    elif single_year and single_make and single_model:
        try:
            year = int(single_year)
        except ValueError:
            context = {
                "request": request,
                "results": None,
                "message": "Invalid year entered.",
                "margin": margin,
                "expenses": expenses,
                "comp_state": comp_state,
                "comp_company": comp_company,
                "comp_runs_drives_only": comp_runs_drives_only,
                "comp_require_keys": comp_require_keys,
                "comp_exclude_parts_only": comp_exclude_parts_only,
                "mileage_range": mileage_range,
                "use_mileage_range": use_mileage_range,
            }
            return templates.TemplateResponse("price_vehicles.html", context)

        make = single_make.strip()
        model = single_model.strip()

        target_mileage = None
        if single_mileage:
            try:
                target_mileage = int(float(single_mileage))
            except Exception:
                target_mileage = None

        filters = build_comp_filters(year, make, model, target_mileage)

        stat = db.query(
            func.avg(models.VehicleSold.sale_price).label("avg"),
            func.count(models.VehicleSold.id).label("cnt"),
        ).filter(*filters).one()

        if stat.cnt == 0 or stat.avg is None:
            results.append({
                "vin": "",
                "year": year,
                "make": make,
                "model": model,
                "avg_sale_price": "No comps",
                "recommended_price": "N/A",
                "comp_count": 0,
            })
        else:
            avg = float(stat.avg)
            rec = avg - expenses - (avg * (margin / 100))

            results.append({
                "vin": "",
                "year": year,
                "make": make,
                "model": model,
                "avg_sale_price": round(avg, 2),
                "recommended_price": round(rec, 2),
                "comp_count": stat.cnt,
            })

    else:
        # Nothing to work with
        context = {
            "request": request,
            "results": None,
            "message": "Please upload a CSV/Excel file or enter a year, make, and model.",
            "margin": margin,
            "expenses": expenses,
            "comp_state": comp_state,
            "comp_company": comp_company,
            "comp_runs_drives_only": comp_runs_drives_only,
            "comp_require_keys": comp_require_keys,
            "comp_exclude_parts_only": comp_exclude_parts_only,
            "mileage_range": mileage_range,
            "use_mileage_range": use_mileage_range,
        }
        return templates.TemplateResponse("price_vehicles.html", context)

    # If user clicked Export, return CSV instead of HTML
    if action == "export":
        # Build CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            ["VIN", "Year", "Make", "Model", "AvgSalePrice", "CompCount", "RecommendedPrice"]
        )
        for r in results:
            writer.writerow([
                r.get("vin", ""),
                r.get("year", ""),
                r.get("make", ""),
                r.get("model", ""),
                r.get("avg_sale_price", ""),
                r.get("comp_count", ""),
                r.get("recommended_price", ""),
            ])
        output.seek(0)
        headers = {
            "Content-Disposition": "attachment; filename=priced_vehicles.csv"
        }
        return StreamingResponse(output, media_type="text/csv", headers=headers)

    # Otherwise render HTML with results
    context = {
        "request": request,
        "results": results,
        "message": None,
        "margin": margin,
        "expenses": expenses,
        "comp_state": comp_state,
        "comp_company": comp_company,
        "comp_runs_drives_only": comp_runs_drives_only,
        "comp_require_keys": comp_require_keys,
        "comp_exclude_parts_only": comp_exclude_parts_only,
        "mileage_range": mileage_range,
        "use_mileage_range": use_mileage_range,
    }
    return templates.TemplateResponse("price_vehicles.html", context)

