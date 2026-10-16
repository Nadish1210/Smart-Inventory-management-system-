"""backend.py - business logic for SmartStock AI.

Everything here is independent of the UI: validation, inventory operations,
sales, forecasting, barcode generation/decoding, reports and backup/restore.
All functions take a ``state`` mapping (``st.session_state`` in the app, or a
plain dict in tests) that holds the in-memory data.
"""
from __future__ import annotations

import copy
import io
import json
import math
import random
import re
from datetime import date, datetime, timedelta
from xml.sax.saxutils import escape

import pandas as pd

APP_NAME = "SmartStock AI"
BACKUP_VERSION = 1
TS_FMT = "%Y-%m-%d %H:%M:%S"

PRODUCT_COLUMNS = [
    "product_id", "barcode", "name", "category", "purchase_price",
    "selling_price", "quantity", "reorder_level", "supplier",
    "expiry_date", "created_at",
]
SALE_COLUMNS = [
    "sale_id", "timestamp", "product_id", "barcode", "product_name",
    "quantity", "unit_price", "unit_cost", "total", "profit",
]
MOVEMENT_COLUMNS = [
    "movement_id", "timestamp", "product_id", "barcode", "product_name",
    "type", "quantity_change", "quantity_after", "note",
]
SUPPLIER_COLUMNS = [
    "supplier_id", "name", "contact_person", "phone", "email", "address", "notes",
]
MOVEMENT_TYPES = ("INITIAL", "IN", "OUT", "ADJUST")
COUNTER_PREFIX = {"product": "P", "sale": "S", "movement": "M", "supplier": "SUP"}
BARCODE_RE = re.compile(r"^[\x21-\x7E]{1,40}$")


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
def _empty_state() -> dict:
    return {
        "products": [], "sales": [], "movements": [], "suppliers": [],
        "counters": {kind: 0 for kind in COUNTER_PREFIX},
    }


def init_state(state) -> None:
    """Create missing data containers without touching existing data."""
    for key, value in _empty_state().items():
        if key not in state:
            state[key] = value


def reset_state(state) -> None:
    for key, value in _empty_state().items():
        state[key] = value


def _ts(when: datetime | None = None) -> str:
    return (when or datetime.now()).strftime(TS_FMT)


def _next_id(state, kind: str) -> str:
    state["counters"][kind] += 1
    return f"{COUNTER_PREFIX[kind]}{state['counters'][kind]:04d}"


# --------------------------------------------------------------------------
# Validation helpers (raise ValueError with a user-friendly message)
# --------------------------------------------------------------------------
def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _num(value, label: str, maximum: float = 1_000_000_000.0) -> float:
    if isinstance(value, bool) or _blank(value):
        raise ValueError(f"{label} is required.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    if number < 0:
        raise ValueError(f"{label} cannot be negative.")
    if number > maximum:
        raise ValueError(f"{label} is too large.")
    return round(number, 2)


def _int(value, label: str, minimum: int = 0, maximum: int = 1_000_000_000) -> int:
    if isinstance(value, bool) or _blank(value):
        raise ValueError(f"{label} is required.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number.") from None
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{label} must be a whole number.")
    number = int(number)
    if number < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    if number > maximum:
        raise ValueError(f"{label} is too large.")
    return number


def _text(value, label: str, required: bool = False, maxlen: int = 120) -> str:
    text = "" if value is None else str(value).strip()
    if required and not text:
        raise ValueError(f"{label} is required.")
    if len(text) > maxlen:
        raise ValueError(f"{label} must be at most {maxlen} characters.")
    return text


def _date_str(value) -> str:
    if _blank(value):
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise ValueError("Expiry date must use the format YYYY-MM-DD.") from None


def _ts_check(value, label: str) -> str:
    try:
        datetime.strptime(str(value), TS_FMT)
    except ValueError:
        raise ValueError(f"{label} must use the format YYYY-MM-DD HH:MM:SS.") from None
    return str(value)


def validate_barcode_value(value) -> str:
    code = "" if value is None else str(value).strip()
    if not BARCODE_RE.match(code):
        raise ValueError(
            "Barcode must be 1-40 printable ASCII characters with no spaces."
        )
    return code


def stock_status(product: dict) -> str:
    if product["quantity"] <= 0:
        return "Out of stock"
    if product["quantity"] <= product["reorder_level"]:
        return "Low stock"
    return "In stock"


# --------------------------------------------------------------------------
# Lookups
# --------------------------------------------------------------------------
def get_product(state, product_id):
    for product in state["products"]:
        if product["product_id"] == product_id:
            return product
    return None


def find_by_barcode(state, barcode):
    for product in state["products"]:
        if product["barcode"] == barcode:
            return product
    return None


def categories(state) -> list[str]:
    return sorted({p["category"] for p in state["products"]})


def supplier_names(state) -> list[str]:
    return sorted(s["name"] for s in state["suppliers"])


def generate_barcode_value(state) -> str:
    while True:
        code = "".join(random.choices("0123456789", k=12))
        if find_by_barcode(state, code) is None:
            return code


def lookup_barcode(state, raw):
    """Return (status, product): status is found / unknown / invalid / empty."""
    raw = "" if raw is None else str(raw).strip()
    if not raw:
        return "empty", None
    try:
        code = validate_barcode_value(raw)
    except ValueError:
        return "invalid", None
    product = find_by_barcode(state, code)
    return ("found", product) if product else ("unknown", None)


# --------------------------------------------------------------------------
# Movements
# --------------------------------------------------------------------------
def _record_movement(state, product, mtype, change, note, when=None) -> dict:
    """Call AFTER the product quantity has been updated."""
    movement = {
        "movement_id": _next_id(state, "movement"),
        "timestamp": _ts(when),
        "product_id": product["product_id"],
        "barcode": product["barcode"],
        "product_name": product["name"],
        "type": mtype,
        "quantity_change": int(change),
        "quantity_after": int(product["quantity"]),
        "note": note,
    }
    state["movements"].append(movement)
    return movement


# --------------------------------------------------------------------------
# Products
# --------------------------------------------------------------------------
def _clean_product_fields(name, category, purchase_price, selling_price,
                          quantity, reorder_level, supplier, expiry_date) -> dict:
    return {
        "name": _text(name, "Product name", True),
        "category": _text(category, "Category", True, 60),
        "purchase_price": _num(purchase_price, "Purchase price"),
        "selling_price": _num(selling_price, "Selling price"),
        "quantity": _int(quantity, "Quantity"),
        "reorder_level": _int(reorder_level, "Reorder level"),
        "supplier": _text(supplier, "Supplier"),
        "expiry_date": _date_str(expiry_date),
    }


def add_product(state, name, category, purchase_price, selling_price, quantity,
                reorder_level, barcode="", supplier="", expiry_date=None, when=None):
    """Return (ok, message, product)."""
    try:
        fields = _clean_product_fields(name, category, purchase_price, selling_price,
                                       quantity, reorder_level, supplier, expiry_date)
        code = _text(barcode, "Barcode", maxlen=40)
        code = validate_barcode_value(code) if code else generate_barcode_value(state)
        existing = find_by_barcode(state, code)
        if existing:
            raise ValueError(f"Barcode {code} already belongs to '{existing['name']}'.")
        product = {
            "product_id": _next_id(state, "product"),
            "barcode": code,
            **fields,
            "created_at": _ts(when),
        }
        state["products"].append(product)
        if product["quantity"] > 0:
            _record_movement(state, product, "INITIAL", product["quantity"],
                             "Opening stock", when)
        return True, f"Added '{product['name']}' (barcode {code}).", product
    except ValueError as exc:
        return False, str(exc), None


def edit_product(state, product_id, name, category, purchase_price, selling_price,
                 quantity, reorder_level, barcode, supplier="", expiry_date=None):
    product = get_product(state, product_id)
    if product is None:
        return False, "Product not found.", None
    try:
        fields = _clean_product_fields(name, category, purchase_price, selling_price,
                                       quantity, reorder_level, supplier, expiry_date)
        code = validate_barcode_value(barcode)
        other = find_by_barcode(state, code)
        if other and other["product_id"] != product_id:
            raise ValueError(f"Barcode {code} already belongs to '{other['name']}'.")
        old_quantity = product["quantity"]
        product.update(fields)
        product["barcode"] = code
        if product["quantity"] != old_quantity:
            _record_movement(state, product, "ADJUST", product["quantity"] - old_quantity,
                             "Manual adjustment from product edit")
        return True, f"Saved changes to '{product['name']}'.", product
    except ValueError as exc:
        return False, str(exc), None


def delete_product(state, product_id):
    product = get_product(state, product_id)
    if product is None:
        return False, "Product not found."
    state["products"].remove(product)
    return True, (f"Deleted '{product['name']}'. Past sales and stock movements "
                  "are kept for reporting.")


def search_products(state, query="", category=None, status=None) -> pd.DataFrame:
    df = products_df(state)
    if df.empty:
        return df
    query = (query or "").strip().lower()
    if query:
        haystack = (df["name"] + " " + df["barcode"] + " " + df["product_id"] + " "
                    + df["category"] + " " + df["supplier"]).str.lower()
        df = df[haystack.str.contains(query, regex=False)]
    if category:
        df = df[df["category"].isin(category)]
    if status and status != "All":
        df = df[df["status"] == status]
    return df


# --------------------------------------------------------------------------
# Stock IN and sales
# --------------------------------------------------------------------------
def stock_in(state, product_id, quantity, note="", when=None):
    """Add received stock to an EXISTING product. Return (ok, message, movement)."""
    product = get_product(state, product_id)
    if product is None:
        return False, "Product not found. Add it first on the Add Product page.", None
    try:
        qty = _int(quantity, "Received quantity", minimum=1)
        note = _text(note, "Delivery note", maxlen=500)
    except ValueError as exc:
        return False, str(exc), None
    product["quantity"] += qty
    movement = _record_movement(state, product, "IN", qty, note or "Stock received", when)
    return True, (f"Received {qty} unit(s) of '{product['name']}'. "
                  f"New stock: {product['quantity']}."), movement


def record_sale(state, product_id, quantity, when=None):
    """Sell units of a product. Return (ok, message, sale)."""
    product = get_product(state, product_id)
    if product is None:
        return False, "Product not found.", None
    try:
        qty = _int(quantity, "Quantity sold", minimum=1)
    except ValueError as exc:
        return False, str(exc), None
    if qty > product["quantity"]:
        return False, (f"Not enough stock: only {product['quantity']} unit(s) of "
                       f"'{product['name']}' available, cannot sell {qty}."), None
    price, cost = product["selling_price"], product["purchase_price"]
    total = round(qty * price, 2)
    profit = round(total - round(qty * cost, 2), 2)
    sale = {
        "sale_id": _next_id(state, "sale"),
        "timestamp": _ts(when),
        "product_id": product["product_id"],
        "barcode": product["barcode"],
        "product_name": product["name"],
        "quantity": qty,
        "unit_price": price,
        "unit_cost": cost,
        "total": total,
        "profit": profit,
    }
    product["quantity"] -= qty
    state["sales"].append(sale)
    _record_movement(state, product, "OUT", -qty, f"Sale {sale['sale_id']}", when)
    return True, (f"Sold {qty} x '{product['name']}' for {total:,.2f} "
                  f"(profit {profit:,.2f})."), sale


# --------------------------------------------------------------------------
# Suppliers
# --------------------------------------------------------------------------
def _clean_supplier(state, name, contact_person, phone, email, address, notes,
                    exclude_id=None) -> dict:
    name = _text(name, "Supplier name", True)
    for other in state["suppliers"]:
        if other["supplier_id"] != exclude_id and other["name"].lower() == name.lower():
            raise ValueError(f"A supplier named '{other['name']}' already exists.")
    email = _text(email, "Email", maxlen=120)
    if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise ValueError("Email address looks invalid.")
    return {
        "name": name,
        "contact_person": _text(contact_person, "Contact person"),
        "phone": _text(phone, "Phone", maxlen=40),
        "email": email,
        "address": _text(address, "Address", maxlen=250),
        "notes": _text(notes, "Notes", maxlen=1000),
    }


def add_supplier(state, name, contact_person="", phone="", email="", address="", notes=""):
    try:
        fields = _clean_supplier(state, name, contact_person, phone, email, address, notes)
    except ValueError as exc:
        return False, str(exc), None
    supplier = {"supplier_id": _next_id(state, "supplier"), **fields}
    state["suppliers"].append(supplier)
    return True, f"Added supplier '{supplier['name']}'.", supplier


def edit_supplier(state, supplier_id, name, contact_person="", phone="", email="",
                  address="", notes=""):
    supplier = next((s for s in state["suppliers"] if s["supplier_id"] == supplier_id), None)
    if supplier is None:
        return False, "Supplier not found.", None
    try:
        fields = _clean_supplier(state, name, contact_person, phone, email, address,
                                 notes, exclude_id=supplier_id)
    except ValueError as exc:
        return False, str(exc), None
    old_name = supplier["name"]
    supplier.update(fields)
    if old_name != supplier["name"]:
        for product in state["products"]:
            if product["supplier"] == old_name:
                product["supplier"] = supplier["name"]
    return True, f"Saved changes to '{supplier['name']}'.", supplier


def delete_supplier(state, supplier_id):
    supplier = next((s for s in state["suppliers"] if s["supplier_id"] == supplier_id), None)
    if supplier is None:
        return False, "Supplier not found."
    state["suppliers"].remove(supplier)
    cleared = 0
    for product in state["products"]:
        if product["supplier"] == supplier["name"]:
            product["supplier"] = ""
            cleared += 1
    return True, f"Deleted '{supplier['name']}'. Cleared the supplier on {cleared} product(s)."


# --------------------------------------------------------------------------
# DataFrames and metrics
# --------------------------------------------------------------------------
def products_df(state) -> pd.DataFrame:
    df = pd.DataFrame(state["products"], columns=PRODUCT_COLUMNS)
    df["status"] = [stock_status(p) for p in state["products"]]
    return df


def sales_df(state) -> pd.DataFrame:
    return pd.DataFrame(state["sales"], columns=SALE_COLUMNS)


def movements_df(state) -> pd.DataFrame:
    return pd.DataFrame(state["movements"], columns=MOVEMENT_COLUMNS)


def suppliers_df(state) -> pd.DataFrame:
    return pd.DataFrame(state["suppliers"], columns=SUPPLIER_COLUMNS)


def filter_by_date(df: pd.DataFrame, start, end, column="timestamp") -> pd.DataFrame:
    if df.empty or start is None or end is None:
        return df
    days = pd.to_datetime(df[column]).dt.date
    return df[(days >= start) & (days <= end)]


def dashboard_metrics(state) -> dict:
    products, sales = state["products"], state["sales"]
    revenue = round(sum(s["total"] for s in sales), 2)
    profit = round(sum(s["profit"] for s in sales), 2)
    return {
        "total_products": len(products),
        "total_units": sum(p["quantity"] for p in products),
        "valuation": round(sum(p["quantity"] * p["purchase_price"] for p in products), 2),
        "revenue": revenue,
        "profit": profit,
        "margin": round(profit / revenue * 100, 1) if revenue else 0.0,
        "out_of_stock": sum(1 for p in products if p["quantity"] <= 0),
        "low_stock": sum(1 for p in products if 0 < p["quantity"] <= p["reorder_level"]),
        "sales_count": len(sales),
    }


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def inventory_report(state) -> pd.DataFrame:
    df = products_df(state)
    df["stock_value_cost"] = (df["quantity"] * df["purchase_price"]).round(2)
    df["stock_value_retail"] = (df["quantity"] * df["selling_price"]).round(2)
    return df[["product_id", "barcode", "name", "category", "supplier", "quantity",
               "reorder_level", "purchase_price", "selling_price", "stock_value_cost",
               "stock_value_retail", "expiry_date", "status"]]


def low_stock_report(state) -> pd.DataFrame:
    df = products_df(state)
    df = df[df["status"] != "In stock"].copy()
    df["units_below_reorder"] = df["reorder_level"] - df["quantity"]
    df = df.sort_values(["quantity", "name"])
    return df[["product_id", "barcode", "name", "category", "supplier", "quantity",
               "reorder_level", "units_below_reorder", "status"]]


def profit_summary(state, start=None, end=None):
    sales = filter_by_date(sales_df(state), start, end)
    revenue = round(float(sales["total"].sum()), 2)
    profit = round(float(sales["profit"].sum()), 2)
    summary = {
        "transactions": int(len(sales)),
        "units_sold": int(sales["quantity"].sum()),
        "revenue": revenue,
        "cost_of_goods": round(revenue - profit, 2),
        "gross_profit": profit,
        "margin_pct": round(profit / revenue * 100, 1) if revenue else 0.0,
    }
    columns = ["product_id", "product_name", "units_sold", "revenue", "cost", "profit"]
    if sales.empty:
        return summary, pd.DataFrame(columns=columns)
    by_product = sales.groupby(["product_id", "product_name"], as_index=False).agg(
        units_sold=("quantity", "sum"), revenue=("total", "sum"), profit=("profit", "sum"))
    by_product["cost"] = (by_product["revenue"] - by_product["profit"]).round(2)
    by_product["revenue"] = by_product["revenue"].round(2)
    by_product["profit"] = by_product["profit"].round(2)
    return summary, by_product[columns].sort_values("profit", ascending=False)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _pdf_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    return "" if value is None else str(value)


def build_pdf_report(title: str, df: pd.DataFrame, summary: list[str] | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=7, leading=9)
    head = ParagraphStyle("head", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
    buffer = io.BytesIO()
    page = landscape(A4)
    doc = SimpleDocTemplate(buffer, pagesize=page, leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title=title)
    story = [Paragraph(escape(title), styles["Title"]),
             Paragraph(f"{APP_NAME} - generated {_ts()}", styles["Normal"]),
             Spacer(1, 6)]
    for line in summary or []:
        story.append(Paragraph(escape(line), styles["Normal"]))
    story.append(Spacer(1, 8))
    if df.empty:
        story.append(Paragraph("No records.", styles["Normal"]))
    else:
        header = [Paragraph(escape(str(c).replace("_", " ").title()), head) for c in df.columns]
        rows = [[Paragraph(escape(_pdf_cell(v)), cell) for v in row]
                for row in df.itertuples(index=False)]
        width = (page[0] - 24 * mm) / len(df.columns)
        table = Table([header] + rows, repeatRows=1, colWidths=[width] * len(df.columns))
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e5f")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f5")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c4ca")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
    doc.build(story)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Barcodes
# --------------------------------------------------------------------------
def generate_barcode_png(value: str, include_text: bool = True) -> bytes:
    """Code 128 barcode as PNG bytes."""
    from barcode import Code128
    from barcode.writer import ImageWriter

    code = validate_barcode_value(value)
    options = {"module_width": 0.35, "module_height": 14.0, "quiet_zone": 5.0,
               "font_size": 9, "text_distance": 4.0, "write_text": include_text}
    try:
        buffer = io.BytesIO()
        Code128(code, writer=ImageWriter()).write(buffer, options=options)
    except Exception:
        buffer = io.BytesIO()
        options["write_text"] = False
        Code128(code, writer=ImageWriter()).write(buffer, options=options)
    return buffer.getvalue()


def _load_font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_label_png(name: str, barcode_value: str, price: float, currency: str = "$") -> bytes:
    """Printable label: product name, barcode and price."""
    from PIL import Image, ImageDraw

    bars = Image.open(io.BytesIO(generate_barcode_png(barcode_value))).convert("RGB")
    pad = 24
    width = max(bars.width + 2 * pad, 640)
    title_font, price_font = _load_font(34), _load_font(40)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    title = name
    while probe.textlength(title, font=title_font) > width - 2 * pad and len(title) > 1:
        title = title[:-2].rstrip("\u2026 ") + "\u2026"
    height = pad + 48 + bars.height + 10 + 56 + pad
    label = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(label)
    draw.text((pad, pad), title, fill="black", font=title_font)
    label.paste(bars, ((width - bars.width) // 2, pad + 48))
    price_text = f"{currency}{price:,.2f}"
    price_w = probe.textlength(price_text, font=price_font)
    draw.text(((width - price_w) / 2, pad + 48 + bars.height + 10), price_text,
              fill="black", font=price_font)
    draw.rectangle([0, 0, width - 1, height - 1], outline="#999999", width=2)
    out = io.BytesIO()
    label.save(out, format="PNG")
    return out.getvalue()


def labels_pdf(label_png: bytes, copies: int) -> bytes:
    """A4 sheet (3 x 8 grid) with ``copies`` repeats of one label."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    copies = max(1, min(int(copies), 240))
    page_w, page_h = A4
    margin, cols, rows = 10 * mm, 3, 8
    cell_w, cell_h = (page_w - 2 * margin) / cols, (page_h - 2 * margin) / rows
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setTitle("Barcode labels")
    reader = ImageReader(io.BytesIO(label_png))
    per_page = cols * rows
    for i in range(copies):
        if i and i % per_page == 0:
            pdf.showPage()
        slot = i % per_page
        col, row = slot % cols, slot // cols
        x = margin + col * cell_w
        y = page_h - margin - (row + 1) * cell_h
        pdf.setStrokeColorRGB(0.8, 0.8, 0.8)
        pdf.rect(x, y, cell_w, cell_h)
        pdf.drawImage(reader, x + 2 * mm, y + 2 * mm, cell_w - 4 * mm, cell_h - 4 * mm,
                      preserveAspectRatio=True, anchor="c")
    pdf.save()
    return buffer.getvalue()


def decode_barcode_image(image_bytes: bytes):
    """Return (codes, error). ``codes`` is a list of {'text', 'format'} dicts."""
    try:
        import zxingcpp
        from PIL import Image, ImageOps
    except ImportError:
        return [], "Barcode decoding library (zxing-cpp) is not installed."
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    except Exception:
        return [], "That file could not be read as an image."
    try:
        attempts = [image, ImageOps.autocontrast(ImageOps.grayscale(image))]
        big = image.resize((image.width * 2, image.height * 2))
        attempts.append(ImageOps.autocontrast(ImageOps.grayscale(big)))
        for candidate in attempts:
            results = zxingcpp.read_barcodes(candidate)
            if results:
                return [{"text": r.text, "format": str(r.format).split(".")[-1]}
                        for r in results], None
    except Exception as exc:  # decoder failures must never crash the UI
        return [], f"Could not decode the image: {exc}"
    return [], None


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------
def forecast_table(state, horizon_days: int, lookback_days: int = 30) -> pd.DataFrame:
    """BASIC estimate: average daily sales x horizon. Not a trained ML model."""
    horizon_days = max(1, int(horizon_days))
    lookback_days = max(1, int(lookback_days))
    columns = ["product_id", "name", "stock", "reorder_level", "units_sold_in_window",
               "avg_daily_sales", "expected_demand", "days_of_stock_left",
               "suggested_reorder_qty", "outlook"]
    if not state["products"]:
        return pd.DataFrame(columns=columns)
    today = datetime.now().date()
    sales = sales_df(state)
    window = lookback_days
    sold = pd.Series(dtype=float)
    if not sales.empty:
        days = pd.to_datetime(sales["timestamp"]).dt.date
        window = max(1, min(lookback_days, (today - days.min()).days + 1))
        recent = sales[days >= today - timedelta(days=window - 1)]
        sold = recent.groupby("product_id")["quantity"].sum()
    rows = []
    for p in state["products"]:
        units = int(sold.get(p["product_id"], 0))
        avg = units / window
        expected = avg * horizon_days
        left = p["quantity"] / avg if avg > 0 else None
        suggested = max(0, math.ceil(expected + p["reorder_level"] - p["quantity"] - 1e-9))
        if expected > p["quantity"]:
            outlook = "Possible shortage"
        elif p["quantity"] - expected <= p["reorder_level"]:
            outlook = "Reorder soon"
        else:
            outlook = "OK"
        rows.append({
            "product_id": p["product_id"], "name": p["name"], "stock": p["quantity"],
            "reorder_level": p["reorder_level"], "units_sold_in_window": units,
            "avg_daily_sales": round(avg, 2), "expected_demand": round(expected, 1),
            "days_of_stock_left": None if left is None else round(left, 1),
            "suggested_reorder_qty": suggested, "outlook": outlook,
        })
    df = pd.DataFrame(rows, columns=columns)
    order = {"Possible shortage": 0, "Reorder soon": 1, "OK": 2}
    return df.sort_values(by="outlook", key=lambda s: s.map(order), kind="stable")


def ml_forecast(state, product_id, horizon_days: int, min_days: int = 21,
                min_selling_days: int = 7) -> dict:
    """Trained model: linear regression on trend + weekday pattern of daily sales."""
    product = get_product(state, product_id)
    if product is None:
        return {"ok": False, "message": "Product not found."}
    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return {"ok": False, "message": "scikit-learn is not installed, so ML forecasting is unavailable."}
    sales = sales_df(state)
    sales = sales[sales["product_id"] == product_id]
    if sales.empty:
        return {"ok": False, "message": "No sales recorded for this product yet."}
    sales = sales.assign(day=pd.to_datetime(sales["timestamp"]).dt.normalize())
    daily = sales.groupby("day")["quantity"].sum()
    today = pd.Timestamp(datetime.now().date())
    end, first = today - pd.Timedelta(days=1), daily.index.min()
    if first > end:
        return {"ok": False, "message": "All sales for this product happened today; more history is needed."}
    index = pd.date_range(first, end, freq="D")
    series = daily.reindex(index, fill_value=0).astype(float)
    selling_days = int((series > 0).sum())
    if len(series) < min_days or selling_days < min_selling_days:
        return {"ok": False, "message": (
            f"Not enough history to train a model: {len(series)} day(s) with "
            f"{selling_days} selling day(s). Need at least {min_days} days and "
            f"{min_selling_days} selling days. Use the basic estimate instead.")}
    origin = index[0]

    def features(dates):
        t = np.array([(d - origin).days for d in dates], dtype=float).reshape(-1, 1)
        weekday = np.eye(7)[[d.weekday() for d in dates]]
        return np.hstack([t, weekday])

    x, y = features(index), series.to_numpy()
    mae = baseline = None
    if len(series) >= 28:
        holdout = LinearRegression().fit(x[:-7], y[:-7])
        predicted = np.clip(holdout.predict(x[-7:]), 0, None)
        mae = float(np.mean(np.abs(predicted - y[-7:])))
        baseline = float(np.mean(np.abs(y[:-7].mean() - y[-7:])))
    model = LinearRegression().fit(x, y)
    future = pd.date_range(today, periods=max(1, int(horizon_days)), freq="D")
    forecast = np.clip(model.predict(features(future)), 0, None)
    total = float(forecast.sum())
    return {
        "ok": True,
        "product": product["name"],
        "history": pd.DataFrame({"date": index, "units": y}),
        "forecast": pd.DataFrame({"date": future, "units": np.round(forecast, 2)}),
        "total_units": round(total, 1),
        "suggested_reorder_qty": max(0, math.ceil(total + product["reorder_level"] - product["quantity"] - 1e-9)),
        "mae": mae, "baseline_mae": baseline,
        "training_days": len(series), "selling_days": selling_days,
    }


# --------------------------------------------------------------------------
# Backup and restore
# --------------------------------------------------------------------------
def export_backup(state) -> dict:
    return {
        "app": APP_NAME,
        "version": BACKUP_VERSION,
        "exported_at": _ts(),
        "products": copy.deepcopy(state["products"]),
        "sales": copy.deepcopy(state["sales"]),
        "movements": copy.deepcopy(state["movements"]),
        "suppliers": copy.deepcopy(state["suppliers"]),
    }


def backup_to_json(state) -> str:
    return json.dumps(export_backup(state), indent=2, ensure_ascii=False)


def parse_backup_bytes(raw: bytes):
    """Return (data, error)."""
    try:
        return json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "File is not valid JSON."


def _max_suffix(ids) -> int:
    best = 0
    for value in ids:
        match = re.search(r"(\d+)$", str(value))
        if match:
            best = max(best, int(match.group(1)))
    return best


def validate_backup(data):
    """Return (cleaned, errors). ``cleaned`` is None when there are errors."""
    if not isinstance(data, dict):
        return None, ["Backup must be a JSON object."]
    errors: list[str] = []
    if data.get("app") != APP_NAME:
        errors.append(f"Not a {APP_NAME} backup (missing app marker).")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or not 1 <= version <= BACKUP_VERSION:
        errors.append(f"Unsupported backup version: {version!r}.")
    for key in ("products", "sales", "movements", "suppliers"):
        if not isinstance(data.get(key), list):
            errors.append(f"'{key}' must be a list.")
    if errors:
        return None, errors

    cleaned = {"products": [], "sales": [], "movements": [], "suppliers": []}

    def check(kind, items, builder):
        seen = set()
        for number, rec in enumerate(items, 1):
            try:
                if not isinstance(rec, dict):
                    raise ValueError("record must be an object")
                clean = builder(rec)
                key = clean[f"{kind}_id"]
                if key in seen:
                    raise ValueError(f"duplicate id {key}")
                seen.add(key)
                cleaned[f"{kind}s"].append(clean)
            except ValueError as exc:
                errors.append(f"{kind.title()} #{number}: {exc}")

    def build_product(r):
        fields = _clean_product_fields(r.get("name"), r.get("category"), r.get("purchase_price"),
                                       r.get("selling_price"), r.get("quantity"),
                                       r.get("reorder_level"), r.get("supplier"), r.get("expiry_date"))
        return {"product_id": _text(r.get("product_id"), "product_id", True, 30),
                "barcode": validate_barcode_value(r.get("barcode")), **fields,
                "created_at": _ts_check(r.get("created_at"), "created_at")}

    def build_sale(r):
        qty = _int(r.get("quantity"), "quantity", 1)
        price, cost = _num(r.get("unit_price"), "unit_price"), _num(r.get("unit_cost"), "unit_cost")
        total = _num(r.get("total"), "total")
        profit = r.get("profit")
        try:
            profit = round(float(profit), 2)
        except (TypeError, ValueError):
            raise ValueError("profit must be a number") from None
        if abs(total - qty * price) > 0.011:
            raise ValueError("total does not equal quantity x unit_price")
        if abs(profit - (total - qty * cost)) > 0.011:
            raise ValueError("profit does not equal total - quantity x unit_cost")
        return {"sale_id": _text(r.get("sale_id"), "sale_id", True, 30),
                "timestamp": _ts_check(r.get("timestamp"), "timestamp"),
                "product_id": _text(r.get("product_id"), "product_id", True, 30),
                "barcode": _text(r.get("barcode"), "barcode", False, 40),
                "product_name": _text(r.get("product_name"), "product_name", True),
                "quantity": qty, "unit_price": price, "unit_cost": cost,
                "total": total, "profit": profit}

    def build_movement(r):
        mtype = r.get("type")
        if mtype not in MOVEMENT_TYPES:
            raise ValueError(f"type must be one of {', '.join(MOVEMENT_TYPES)}")
        change = r.get("quantity_change")
        if isinstance(change, bool) or not isinstance(change, (int, float)) or change != int(change):
            raise ValueError("quantity_change must be a whole number")
        return {"movement_id": _text(r.get("movement_id"), "movement_id", True, 30),
                "timestamp": _ts_check(r.get("timestamp"), "timestamp"),
                "product_id": _text(r.get("product_id"), "product_id", True, 30),
                "barcode": _text(r.get("barcode"), "barcode", False, 40),
                "product_name": _text(r.get("product_name"), "product_name", True),
                "type": mtype, "quantity_change": int(change),
                "quantity_after": _int(r.get("quantity_after"), "quantity_after"),
                "note": _text(r.get("note"), "note", False, 500)}

    def build_supplier(r):
        email = _text(r.get("email"), "email")
        return {"supplier_id": _text(r.get("supplier_id"), "supplier_id", True, 30),
                "name": _text(r.get("name"), "name", True),
                "contact_person": _text(r.get("contact_person"), "contact_person"),
                "phone": _text(r.get("phone"), "phone", False, 40), "email": email,
                "address": _text(r.get("address"), "address", False, 250),
                "notes": _text(r.get("notes"), "notes", False, 1000)}

    check("product", data["products"], build_product)
    check("sale", data["sales"], build_sale)
    check("movement", data["movements"], build_movement)
    check("supplier", data["suppliers"], build_supplier)

    barcodes: dict[str, str] = {}
    for p in cleaned["products"]:
        if p["barcode"] in barcodes:
            errors.append(f"Barcode {p['barcode']} is used by both {barcodes[p['barcode']]} and {p['product_id']}.")
        barcodes[p["barcode"]] = p["product_id"]
    last_after: dict[str, int] = {}
    for m in cleaned["movements"]:
        last_after[m["product_id"]] = m["quantity_after"]
    for p in cleaned["products"]:
        if p["product_id"] in last_after and last_after[p["product_id"]] != p["quantity"]:
            errors.append(f"Product {p['product_id']}: quantity {p['quantity']} does not match "
                          f"its last stock movement ({last_after[p['product_id']]}).")
    if errors:
        return None, errors[:25]
    return cleaned, []


def backup_summary(cleaned: dict) -> str:
    return (f"{len(cleaned['products'])} products, {len(cleaned['sales'])} sales, "
            f"{len(cleaned['movements'])} stock movements, {len(cleaned['suppliers'])} suppliers")


def restore_backup(state, data):
    """Replace ALL current data with a validated backup. Return (ok, message)."""
    cleaned, errors = validate_backup(data)
    if errors:
        return False, "Backup rejected: " + "; ".join(errors[:5])
    state["products"] = cleaned["products"]
    state["sales"] = cleaned["sales"]
    state["movements"] = cleaned["movements"]
    state["suppliers"] = cleaned["suppliers"]
    state["counters"] = {
        "product": _max_suffix(p["product_id"] for p in cleaned["products"]),
        "sale": _max_suffix(s["sale_id"] for s in cleaned["sales"]),
        "movement": _max_suffix(m["movement_id"] for m in cleaned["movements"]),
        "supplier": _max_suffix(s["supplier_id"] for s in cleaned["suppliers"]),
    }
    return True, f"Restored {backup_summary(cleaned)}."


# --------------------------------------------------------------------------
# Sample data (for trying the app and testing forecasts)
# --------------------------------------------------------------------------
def load_sample_data(state, days: int = 35, seed: int = 7) -> str:
    """Replace all data with a consistent demo store and back-dated sales."""
    rng = random.Random(seed)
    reset_state(state)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start = today - timedelta(days=days)
    for name, contact in [("FreshFarm Dairy", "Ana Ruiz"), ("Metro Wholesale", "Sam Lee"),
                          ("HomeCare Supplies", "Priya Nair")]:
        add_supplier(state, name, contact, "555-0100", "", "", "Sample supplier")
    catalog = [
        ("Whole Milk 1L", "Dairy", 0.80, 1.20, 6, 14, 20, "FreshFarm Dairy", 9),
        ("Brown Bread", "Bakery", 1.00, 1.80, 5, 4, 10, "Metro Wholesale", 3),
        ("Basmati Rice 5kg", "Grocery", 6.50, 9.99, 2, 40, 10, "Metro Wholesale", None),
        ("Orange Juice 1L", "Beverages", 1.40, 2.50, 4, 0, 12, "Metro Wholesale", 30),
        ("Dark Chocolate Bar", "Snacks", 0.90, 1.99, 7, 60, 25, "Metro Wholesale", 180),
        ("Dish Soap 500ml", "Household", 1.10, 2.20, 2, 18, 8, "HomeCare Supplies", None),
        ("AA Batteries 4-pack", "Household", 2.30, 4.50, 1, 9, 5, "HomeCare Supplies", None),
        ("Free-range Eggs 12", "Dairy", 2.20, 3.60, 5, 7, 15, "FreshFarm Dairy", 14),
    ]
    for name, category, cost, price, rate, final_qty, reorder, supplier, shelf in catalog:
        plan = []
        for offset in range(days):
            if rng.random() < 0.8:
                qty = max(1, round(rng.gauss(rate, max(1.0, rate * 0.4))))
                if (start + timedelta(days=offset)).weekday() >= 5:
                    qty = round(qty * 1.5)
                plan.append((offset, qty))
        initial = sum(q for _, q in plan) + final_qty
        expiry = (today + timedelta(days=shelf)).date() if shelf else None
        ok, msg, product = add_product(state, name, category, cost, price, initial, reorder,
                                       supplier=supplier, expiry_date=expiry, when=start)
        if not ok:
            raise RuntimeError(msg)
        for offset, qty in plan:
            when = start + timedelta(days=offset, hours=rng.randint(9, 19), minutes=rng.randint(0, 59))
            ok, msg, _ = record_sale(state, product["product_id"], qty, when=when)
            if not ok:
                raise RuntimeError(msg)
    return f"Loaded sample data: {len(state['products'])} products, {len(state['sales'])} sales."
