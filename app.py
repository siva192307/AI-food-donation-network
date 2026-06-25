# =============================================================================
# AI Food Donation Network - Flask Application  (v4 - Full Analytics + NGO)
# =============================================================================

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import sqlite3
import os
import json
from datetime import datetime, date
from collections import defaultdict
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = "food_donation_secret_key_2024"

DATABASE = os.path.join(app.instance_path, "donations.db")

# ---------------------------------------------------------------------------
# ML Model — loaded once at startup
# ---------------------------------------------------------------------------
_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "food_quality_model.pkl")
_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")

def _load_ml_assets():
    try:
        return joblib.load(_MODEL_PATH), joblib.load(_ENCODER_PATH)
    except FileNotFoundError:
        return None, None

_rf_model, _label_encoder = _load_ml_assets()

# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------

def get_db_connection():
    """Open a new SQLite connection with row_factory for dict-like access."""
    os.makedirs(app.instance_path, exist_ok=True)
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all application tables if they do not already exist."""
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS donations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            donor_name  TEXT    NOT NULL,
            email       TEXT    NOT NULL,
            phone       TEXT    NOT NULL,
            food_type   TEXT    NOT NULL,
            quantity    TEXT    NOT NULL,
            expiry_date TEXT    NOT NULL,
            location    TEXT    NOT NULL,
            notes       TEXT,
            status      TEXT    NOT NULL DEFAULT 'Pending',
            ngo_id      INTEGER REFERENCES ngos(id),
            created_at  TEXT    NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ngos (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ngo_name       TEXT    NOT NULL,
            location       TEXT    NOT NULL,
            contact_person TEXT    NOT NULL,
            phone          TEXT    NOT NULL,
            email          TEXT    NOT NULL,
            capacity       INTEGER NOT NULL DEFAULT 0,
            registered_at  TEXT    NOT NULL
        )
    """)

    # donation_assignments table — tracks full assignment lifecycle
    conn.execute("""
        CREATE TABLE IF NOT EXISTS donation_assignments (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            donation_id   INTEGER NOT NULL REFERENCES donations(id),
            ngo_id        INTEGER NOT NULL REFERENCES ngos(id),
            assigned_date TEXT    NOT NULL,
            status        TEXT    NOT NULL DEFAULT 'Pending'
        )
    """)

    # Safe migrations for columns added in later versions
    for migration in [
        "ALTER TABLE donations ADD COLUMN ngo_id INTEGER REFERENCES ngos(id)",
    ]:
        try:
            conn.execute(migration)
        except Exception:
            pass

    conn.commit()
    conn.close()


with app.app_context():
    init_db()

# ---------------------------------------------------------------------------
# Helper: parse numeric quantity from free-text ("10 kg" → 10.0)
# ---------------------------------------------------------------------------
def _parse_qty(raw: str) -> float:
    """Extract the first numeric token from a quantity string."""
    import re
    m = re.search(r"[\d.]+", str(raw))
    return float(m.group()) if m else 0.0


# ===========================================================================
# HOME PAGE
# ===========================================================================

@app.route("/")
def index():
    conn = get_db_connection()

    total_donations = conn.execute("SELECT COUNT(*) FROM donations").fetchone()[0]
    pending         = conn.execute("SELECT COUNT(*) FROM donations WHERE status='Pending'").fetchone()[0]
    collected       = conn.execute("SELECT COUNT(*) FROM donations WHERE status='Collected'").fetchone()[0]
    distributed     = conn.execute("SELECT COUNT(*) FROM donations WHERE status='Distributed'").fetchone()[0]
    total_ngos      = conn.execute("SELECT COUNT(*) FROM ngos").fetchone()[0]
    recent          = conn.execute("SELECT * FROM donations ORDER BY id DESC LIMIT 4").fetchall()

    # Impact metrics ─────────────────────────────────────────────────────────
    rows = conn.execute("SELECT quantity FROM donations").fetchall()
    conn.close()

    total_kg = sum(_parse_qty(r["quantity"]) for r in rows)
    meals_served     = int(total_kg * 2)
    co2_prevented    = round(total_kg * 2.5, 1)

    return render_template(
        "index.html",
        total_donations=total_donations,
        pending=pending,
        collected=collected,
        distributed=distributed,
        total_ngos=total_ngos,
        total_kg=round(total_kg, 1),
        meals_served=meals_served,
        co2_prevented=co2_prevented,
        recent_donations=recent,
    )


# ===========================================================================
# DONATION FORM
# ===========================================================================

@app.route("/donate", methods=["GET", "POST"])
def donate():
    if request.method == "POST":
        donor_name  = request.form.get("donor_name",  "").strip()
        email       = request.form.get("email",        "").strip()
        phone       = request.form.get("phone",        "").strip()
        food_type   = request.form.get("food_type",    "").strip()
        quantity    = request.form.get("quantity",     "").strip()
        expiry_date = request.form.get("expiry_date",  "").strip()
        location    = request.form.get("location",     "").strip()
        notes       = request.form.get("notes",        "").strip()

        if not all([donor_name, email, phone, food_type, quantity, expiry_date, location]):
            flash("All required fields must be filled in.", "danger")
            return redirect(url_for("donate"))

        conn = get_db_connection()
        conn.execute(
            """INSERT INTO donations
               (donor_name, email, phone, food_type, quantity,
                expiry_date, location, notes, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Pending', ?)""",
            (donor_name, email, phone, food_type, quantity,
             expiry_date, location, notes,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        conn.close()

        flash("Thank you! Your donation has been submitted successfully.", "success")
        return redirect(url_for("dashboard"))

    return render_template("donate.html")


# ===========================================================================
# DASHBOARD  (with Chart.js analytics data)
# ===========================================================================

@app.route("/dashboard")
def dashboard():
    conn      = get_db_connection()
    donations = conn.execute("SELECT * FROM donations ORDER BY id DESC").fetchall()
    total       = conn.execute("SELECT COUNT(*) FROM donations").fetchone()[0]
    pending     = conn.execute("SELECT COUNT(*) FROM donations WHERE status='Pending'").fetchone()[0]
    collected   = conn.execute("SELECT COUNT(*) FROM donations WHERE status='Collected'").fetchone()[0]
    distributed = conn.execute("SELECT COUNT(*) FROM donations WHERE status='Distributed'").fetchone()[0]

    # ── Chart 1: Donations by Food Type (Pie) ────────────────────────────
    by_type_rows = conn.execute(
        "SELECT food_type, COUNT(*) as cnt FROM donations GROUP BY food_type ORDER BY cnt DESC"
    ).fetchall()
    chart_labels  = [r["food_type"] for r in by_type_rows]
    chart_counts  = [r["cnt"]       for r in by_type_rows]

    # ── Chart 2: Safe vs Unsafe AI Predictions (Doughnut) ────────────────
    # We derive from status: Distributed = Safe-used; Pending/Collected = in-progress
    safe_count   = conn.execute("SELECT COUNT(*) FROM donations WHERE status='Distributed'").fetchone()[0]
    unsafe_count = conn.execute("SELECT COUNT(*) FROM donations WHERE status='Pending'").fetchone()[0]
    collected_ct = collected

    # ── Chart 3: Quantity by Category (Bar) ──────────────────────────────
    qty_rows = conn.execute("SELECT food_type, quantity FROM donations").fetchall()
    qty_by_type = defaultdict(float)
    for r in qty_rows:
        qty_by_type[r["food_type"]] += _parse_qty(r["quantity"])
    bar_labels = list(qty_by_type.keys())
    bar_values = [round(v, 1) for v in qty_by_type.values()]

    # ── Chart 4: Monthly Donation Trend (Line) ────────────────────────────
    monthly_rows = conn.execute(
        """SELECT substr(created_at,1,7) as month, COUNT(*) as cnt
           FROM donations
           GROUP BY month
           ORDER BY month ASC
           LIMIT 12"""
    ).fetchall()
    trend_labels = [r["month"] for r in monthly_rows]
    trend_values = [r["cnt"]   for r in monthly_rows]

    conn.close()

    # Serialize chart data to JSON for the template
    chart_data = {
        "pie":   {"labels": chart_labels,  "data": chart_counts},
        "donut": {"labels": ["Distributed","Pending","Collected"],
                  "data":   [safe_count, unsafe_count, collected_ct]},
        "bar":   {"labels": bar_labels,    "data": bar_values},
        "line":  {"labels": trend_labels,  "data": trend_values},
    }

    return render_template(
        "dashboard.html",
        donations=donations,
        total=total,
        pending=pending,
        collected=collected,
        distributed=distributed,
        now=date.today(),
        chart_data=json.dumps(chart_data),
    )


@app.route("/update_status/<int:donation_id>", methods=["POST"])
def update_status(donation_id):
    new_status = request.form.get("status")
    if new_status in {"Pending", "Collected", "Distributed"}:
        conn = get_db_connection()
        conn.execute("UPDATE donations SET status = ? WHERE id = ?", (new_status, donation_id))
        conn.commit()
        conn.close()
        flash(f"Donation #{donation_id} status updated to '{new_status}'.", "success")
    else:
        flash("Invalid status value.", "danger")
    return redirect(url_for("dashboard"))


@app.route("/delete/<int:donation_id>", methods=["POST"])
def delete_donation(donation_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM donation_assignments WHERE donation_id = ?", (donation_id,))
    conn.execute("DELETE FROM donations WHERE id = ?", (donation_id,))
    conn.commit()
    conn.close()
    flash(f"Donation #{donation_id} has been deleted.", "warning")
    return redirect(url_for("dashboard"))


# ===========================================================================
# NGO MANAGEMENT
# ===========================================================================

@app.route("/ngo/register", methods=["GET", "POST"])
def ngo_register():
    """Add a new NGO (GET = blank form, POST = save)."""
    if request.method == "POST":
        ngo_name       = request.form.get("ngo_name",       "").strip()
        location       = request.form.get("location",       "").strip()
        contact_person = request.form.get("contact_person", "").strip()
        phone          = request.form.get("phone",          "").strip()
        email          = request.form.get("email",          "").strip()
        capacity_raw   = request.form.get("capacity",       "0").strip()

        if not all([ngo_name, location, contact_person, phone, email]):
            flash("All required fields must be filled in.", "danger")
            return redirect(url_for("ngo_register"))

        try:
            capacity = int(capacity_raw)
            if capacity < 0:
                raise ValueError
        except ValueError:
            flash("Capacity must be a positive whole number.", "danger")
            return redirect(url_for("ngo_register"))

        conn = get_db_connection()
        existing = conn.execute("SELECT id FROM ngos WHERE email = ?", (email,)).fetchone()
        if existing:
            conn.close()
            flash("An NGO with this email address is already registered.", "warning")
            return redirect(url_for("ngo_register"))

        conn.execute(
            """INSERT INTO ngos (ngo_name, location, contact_person, phone, email, capacity, registered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ngo_name, location, contact_person, phone, email, capacity,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        conn.close()
        flash(f"NGO '{ngo_name}' registered successfully!", "success")
        return redirect(url_for("ngo_dashboard"))

    return render_template("ngo_register.html", ngo=None)


@app.route("/ngo/edit/<int:ngo_id>", methods=["GET", "POST"])
def ngo_edit(ngo_id):
    """Edit an existing NGO."""
    conn = get_db_connection()
    ngo  = conn.execute("SELECT * FROM ngos WHERE id = ?", (ngo_id,)).fetchone()
    if not ngo:
        conn.close()
        flash("NGO not found.", "danger")
        return redirect(url_for("ngo_dashboard"))

    if request.method == "POST":
        ngo_name       = request.form.get("ngo_name",       "").strip()
        location       = request.form.get("location",       "").strip()
        contact_person = request.form.get("contact_person", "").strip()
        phone          = request.form.get("phone",          "").strip()
        email          = request.form.get("email",          "").strip()
        capacity_raw   = request.form.get("capacity",       "0").strip()

        if not all([ngo_name, location, contact_person, phone, email]):
            conn.close()
            flash("All required fields must be filled in.", "danger")
            return redirect(url_for("ngo_edit", ngo_id=ngo_id))

        try:
            capacity = int(capacity_raw)
        except ValueError:
            conn.close()
            flash("Capacity must be a whole number.", "danger")
            return redirect(url_for("ngo_edit", ngo_id=ngo_id))

        # Allow same email for this NGO but block duplicate with another
        dup = conn.execute(
            "SELECT id FROM ngos WHERE email = ? AND id != ?", (email, ngo_id)
        ).fetchone()
        if dup:
            conn.close()
            flash("Another NGO already uses this email.", "warning")
            return redirect(url_for("ngo_edit", ngo_id=ngo_id))

        conn.execute(
            """UPDATE ngos SET ngo_name=?, location=?, contact_person=?,
               phone=?, email=?, capacity=? WHERE id=?""",
            (ngo_name, location, contact_person, phone, email, capacity, ngo_id),
        )
        conn.commit()
        conn.close()
        flash(f"NGO '{ngo_name}' updated successfully.", "success")
        return redirect(url_for("ngo_dashboard"))

    conn.close()
    return render_template("ngo_register.html", ngo=ngo)


@app.route("/ngo/dashboard")
def ngo_dashboard():
    conn        = get_db_connection()
    search_q    = request.args.get("q", "").strip()

    if search_q:
        ngos = conn.execute(
            """SELECT * FROM ngos
               WHERE ngo_name LIKE ? OR location LIKE ? OR contact_person LIKE ?
               ORDER BY id DESC""",
            (f"%{search_q}%", f"%{search_q}%", f"%{search_q}%"),
        ).fetchall()
    else:
        ngos = conn.execute("SELECT * FROM ngos ORDER BY id DESC").fetchall()

    total_ngos   = conn.execute("SELECT COUNT(*) FROM ngos").fetchone()[0]
    total_don    = conn.execute("SELECT COUNT(*) FROM donations").fetchone()[0]
    active_don   = conn.execute(
        "SELECT COUNT(*) FROM donations WHERE status IN ('Pending','Collected')"
    ).fetchone()[0]
    food_saved   = conn.execute(
        "SELECT COUNT(*) FROM donations WHERE status='Distributed'"
    ).fetchone()[0]
    active_ngos  = conn.execute(
        "SELECT COUNT(DISTINCT ngo_id) FROM donations WHERE ngo_id IS NOT NULL"
    ).fetchone()[0]
    conn.close()

    return render_template(
        "ngo_dashboard.html",
        ngos=ngos,
        total_ngos=total_ngos,
        total_donations=total_don,
        active_donations=active_don,
        food_saved=food_saved,
        active_ngos=active_ngos,
        search_q=search_q,
    )


@app.route("/ngo/delete/<int:ngo_id>", methods=["POST"])
def delete_ngo(ngo_id):
    conn = get_db_connection()
    conn.execute("UPDATE donations SET ngo_id = NULL WHERE ngo_id = ?", (ngo_id,))
    conn.execute("DELETE FROM donation_assignments WHERE ngo_id = ?", (ngo_id,))
    conn.execute("DELETE FROM ngos WHERE id = ?", (ngo_id,))
    conn.commit()
    conn.close()
    flash(f"NGO #{ngo_id} has been removed.", "warning")
    return redirect(url_for("ngo_dashboard"))


@app.route("/assign_ngo", methods=["POST"])
def assign_ngo():
    """Assign a donation to an NGO and create a donation_assignment record."""
    donation_id = request.form.get("donation_id", type=int)
    ngo_id      = request.form.get("ngo_id",      type=int)

    if not donation_id or not ngo_id:
        flash("Invalid assignment — donation or NGO not specified.", "danger")
        return redirect(url_for("predict"))

    conn = get_db_connection()
    donation = conn.execute("SELECT id FROM donations WHERE id = ?", (donation_id,)).fetchone()
    ngo      = conn.execute("SELECT ngo_name FROM ngos WHERE id = ?", (ngo_id,)).fetchone()

    if not donation or not ngo:
        conn.close()
        flash("Donation or NGO not found.", "danger")
        return redirect(url_for("dashboard"))

    conn.execute(
        "UPDATE donations SET ngo_id = ?, status = 'Collected' WHERE id = ?",
        (ngo_id, donation_id),
    )
    # Create assignment record
    conn.execute(
        """INSERT INTO donation_assignments (donation_id, ngo_id, assigned_date, status)
           VALUES (?, ?, ?, 'Pending')""",
        (donation_id, ngo_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()
    flash(f"Donation #{donation_id} successfully assigned to '{ngo['ngo_name']}'.", "success")
    return redirect(url_for("dashboard"))


# ===========================================================================
# AI PREDICTION
# ===========================================================================

_SAFE_ACTIONS   = "Assign to nearest NGO for immediate collection and distribution."
_UNSAFE_ACTIONS = "Discard safely or compost. Do not distribute to recipients."

_RECOMMENDATIONS = {
    ("Yes", "high"):   "Food meets all safety standards. Ready for immediate distribution.",
    ("Yes", "medium"): "Food appears safe. Quick visual inspection recommended before collection.",
    ("Yes", "low"):    "Food may be safe but confidence is moderate. Request volunteer inspection.",
    ("No",  "high"):   "Food does not meet safety standards. Do NOT distribute.",
    ("No",  "medium"): "Food is likely unsafe. Reduce storage time/temperature or discard.",
    ("No",  "low"):    "Model leans unsafe. Manual inspection required before any decision.",
}


def _confidence_band(p: float) -> str:
    return "high" if p >= 76 else ("medium" if p >= 51 else "low")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if _rf_model is None or _label_encoder is None:
        flash("ML model not found. Please run 'python train_model.py' first.", "danger")
        return render_template("predict.html")

    if request.method == "GET":
        return render_template("predict.html")

    try:
        food_type     = request.form["food_type"].strip()
        quantity      = float(request.form["quantity"])
        prep_time     = float(request.form["prep_time"])
        storage_hours = float(request.form["storage_hours"])
        temperature   = float(request.form["temperature"])
    except (KeyError, ValueError):
        flash("Invalid input — please check all fields.", "danger")
        return redirect(url_for("predict"))

    known_classes = list(_label_encoder.classes_)
    if food_type not in known_classes:
        flash(f"Unknown food type. Supported: {', '.join(known_classes)}", "danger")
        return redirect(url_for("predict"))

    food_type_enc = _label_encoder.transform([food_type])[0]
    features      = np.array([[food_type_enc, quantity, prep_time, storage_hours, temperature]])
    raw_pred      = _rf_model.predict(features)[0]
    proba         = _rf_model.predict_proba(features)[0]

    prediction     = "Yes" if raw_pred == 1 else "No"
    confidence     = round(float(proba[raw_pred]) * 100, 1)
    band           = _confidence_band(confidence)
    recommendation = _RECOMMENDATIONS.get((prediction, band), "")
    suggested_action = _SAFE_ACTIONS if prediction == "Yes" else _UNSAFE_ACTIONS

    input_data = dict(food_type=food_type, quantity=quantity,
                      prep_time=prep_time, storage_hours=storage_hours,
                      temperature=temperature)

    available_ngos   = []
    last_donation_id = None
    if prediction == "Yes":
        conn = get_db_connection()
        available_ngos   = conn.execute("SELECT * FROM ngos ORDER BY ngo_name ASC").fetchall()
        row              = conn.execute("SELECT id FROM donations ORDER BY id DESC LIMIT 1").fetchone()
        last_donation_id = row[0] if row else None
        conn.close()

    return render_template(
        "prediction_result.html",
        prediction=prediction,
        confidence=confidence,
        recommendation=recommendation,
        suggested_action=suggested_action,
        input_data=input_data,
        available_ngos=available_ngos,
        last_donation_id=last_donation_id,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
