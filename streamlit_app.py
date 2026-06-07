import streamlit as st
import google.generativeai as genai
import pandas as pd
import base64
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image
import io
import re
import json
import fitz  # PyMuPDF for PDF processing
import plotly.express as px  # Import Plotly for visualization
from pymongo import MongoClient
from difflib import SequenceMatcher  # For semantic-ish matching of field names
import logging

# --------------------------------------------------------------------------LOGGING--------------------------------------------------------------------------
# Production logging so the routing decisions are observable in the container logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("invoice_pipeline")


#--------------------------------------------------------------------------API KEY INITIALIZATIONS--------------------------------------------------------------------------
# Gemini API Key
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
MONGO_URI = st.secrets["MONGO_URI"]  # Store this in Streamlit Secrets
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["officeflow-invoice-petty"]  # Connect to the database
invoice_collection = db["Invoice"]
projects_coll = db["Project"]

# Initialize session state for projects if not already initialized
if "projects" not in st.session_state:
    st.session_state.projects = {}  # Dictionary to store project data


#--------------------------------------------------------------------------HYBRID PIPELINE CONFIG--------------------------------------------------------------------------
# These thresholds control the digital-vs-scanned decision. They are deliberately
# conservative: when in doubt the pipeline falls back to the VISION path, because
# Vision is the safe (if more expensive) route that handles anything.
#
# Tune these against YOUR document mix. The defaults below are sane production
# starting points validated against typical B2B invoices.
DIGITAL_DETECTION_CONFIG = {
    # Minimum meaningful characters per page to even consider a page "digital".
    "min_chars_per_page": 60,
    # Minimum extracted words per page.
    "min_words_per_page": 12,
    # Ratio of "normal" printable chars (latin+arabic+digits+punct) to total.
    # Below this we treat the text layer as garbage/encoded and force OCR.
    "min_normal_char_ratio": 0.55,
    # If a single embedded image covers >= this fraction of the page AND there is
    # little/no text, the page is a scan wrapped in a PDF -> OCR.
    "full_page_image_cover": 0.80,
    # A digital page should have at least this much text relative to its image
    # footprint. Catches "digital template + big logo" false positives.
    "min_text_when_large_image": 25,  # chars
}

# DPI used when rasterising for the Vision path. 200 is a good accuracy/size
# trade-off for Gemini; bump to 300 for dense/handwritten docs if accuracy drags.
RASTER_DPI = 200


#--------------------------------------------------------------------------DYNAMIC FIELD CONFIGURATION--------------------------------------------------------------------------
# Each predefined field has:
#   label   -> how it is shown to the user / asked from the LLM
#   key     -> the canonical snake_case key we store it under (keeps old code working)
#   type    -> string / float / bool / line_items (controls the JSON schema hint)
#   aliases -> spellings the LLM might return, used when reading the value back
PREDEFINED_FIELDS = [
    {"label": "Invoice Number", "key": "Invoice_Number", "type": "string",
     "aliases": ["Invoice Number", "Invoice_Number", "InvoiceNumber", "Invoice No", "Invoice No.",
                 "رقم الفاتورة", "رقم الفاتوره", "No.", "Ref", "Ref No", "Reference No",
                 "Receipt No", "Receipt Number", "Voucher No", "Doc No", "Document No",
                 "Cash No", "Invoice", "Folio No", "Bill No", "Transaction No"]},
    {"label": "Invoice Date", "key": "Invoice_Date", "type": "string",
     "aliases": ["Invoice Date", "Invoice_Date", "InvoiceDate", "Date"]},
    {"label": "Supplier Name", "key": "Supplier_Name", "type": "string",
     "aliases": ["Supplier Name", "Supplier_Name", "SupplierName"]},
    {"label": "Supplier VAT", "key": "Supplier_VAT", "type": "string",
     "aliases": ["Supplier VAT", "Supplier_VAT", "SupplierVAT"]},
    {"label": "Customer Name", "key": "Customer_Name", "type": "string",
     "aliases": ["Customer Name", "Customer_Name", "CustomerName",
                 "Guest Name", "Guest", "Passenger Name", "Passenger",
                 "Employee Name", "Employee", "Purchaser", "Buyer",
                 "Client Name", "Client", "Payee", "Attn", "Contact Person",
                 "Recipient", "Cardholder", "Name"]},
    {"label": "Customer VAT", "key": "Customer_VAT", "type": "string",
     "aliases": ["Customer VAT", "Customer_VAT", "CustomerVAT"]},
    {"label": "Amount Before VAT", "key": "Amount_Before_VAT", "type": "float",
     "aliases": ["Amount Before VAT", "Amount_Before_VAT", "AmountBeforeVAT"]},
    {"label": "VAT Amount", "key": "VAT_Amount", "type": "float",
     "aliases": ["VAT Amount", "VAT_Amount", "VATAmount"]},
    {"label": "Total Amount After VAT", "key": "Total_Amount_After_VAT", "type": "float",
     "aliases": ["Total Amount After VAT", "Total_Amount_After_VAT", "TotalAmountAfterVAT"]},
    {"label": "QR Code Present", "key": "QR_Code_Present", "type": "bool",
     "aliases": ["QR Code Present", "QR_Code_Present", "QRCodePresent"]},
    {"label": "Line Items", "key": "Line_Items", "type": "line_items",
     "aliases": ["Line Items", "Line_Items", "LineItems"]},
]


def normalize_key(key):
    """Lowercase a key and strip everything except letters/digits so that
    'Invoice Number', 'invoice_number' and 'InvoiceNumber' all compare equal."""
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def find_value(invoice_data, candidates, default=None):
    """Read a value out of the LLM's JSON, tolerating wording differences.

    1. Exact match on any of the candidate spellings (after normalizing).
    2. Fuzzy fallback (e.g. 'Invoice No' vs 'Invoice Number') above a threshold.
    """
    norm_map = {normalize_key(k): k for k in invoice_data.keys()}

    # 1) exact (normalized) match against every accepted spelling
    for cand in candidates:
        nk = normalize_key(cand)
        if nk in norm_map:
            return invoice_data[norm_map[nk]]

    # 2) fuzzy fallback against the primary (first) candidate
    primary = normalize_key(candidates[0])
    best_key, best_ratio = None, 0.0
    for nk, original in norm_map.items():
        ratio = SequenceMatcher(None, primary, nk).ratio()
        if ratio > best_ratio:
            best_ratio, best_key = ratio, original
    if best_key and best_ratio >= 0.82:
        return invoice_data[best_key]

    return default


def build_invoice_prompt(selected_fields, custom_fields, source_text=None):
    """Build the Gemini prompt dynamically from the user's chosen fields.

    selected_fields -> list of PREDEFINED_FIELDS dicts the user ticked
    custom_fields   -> list of extra field-name strings the user typed
    source_text     -> OPTIONAL. When provided (digital PDF path), the extracted
                       text layer is appended so Gemini extracts from lossless text
                       instead of (or in addition to) the rendered image.
    """
    schema_lines = []
    bullet_lines = []

    for field in selected_fields:
        if field["type"] == "line_items":
            schema_lines.append(
                '  "Line Items": [\n'
                '    {\n'
                '      "<column header 1>": "<value>",\n'
                '      "<column header 2>": "<value>",\n'
                '      "... (all columns present in the table)": "<value>"\n'
                '    }\n'
                '  ]'
            )
            bullet_lines.append(
                "- Line Items: extract every REAL product/service row from the line items table."
            )
        elif field["type"] == "float":
            schema_lines.append(f'  "{field["label"]}": <float>')
            bullet_lines.append(f"- {field['label']}")
        elif field["type"] == "bool":
            schema_lines.append(f'  "{field["label"]}": <boolean>')
            bullet_lines.append(f"- {field['label']}")
        else:
            schema_lines.append(f'  "{field["label"]}": "<string>"')
            bullet_lines.append(f"- {field['label']}")

    # Custom user-defined fields (default to string, value as it appears on the invoice)
    for cf in custom_fields:
        schema_lines.append(f'  "{cf}": "<string>"')
        bullet_lines.append(f"- {cf}")

    schema = "{\n" + ",\n".join(schema_lines) + "\n}"

    has_line_items = any(f["type"] == "line_items" for f in selected_fields)

    line_items_rules = ""
    if has_line_items:
        line_items_rules = """

LINE ITEMS — DETAILED RULES (read carefully before extracting):

1. SEMANTIC GROUPING — A handwritten or informal invoice may spread a single item
   description across multiple physical lines. Before splitting into separate rows,
   ask yourself: "Do these lines together describe ONE thing being sold?"
   Signals that lines belong to ONE item:
     • There is only ONE quantity / price covering all of them.
     • The second line continues the first (e.g. "50 Ton Garvin" + "Hydraulic Piston Cooling"
       = one item "50 Ton Garvin Hydraulic Piston Cooling").
     • Removing either line would leave the description incomplete.
   Signals that lines are SEPARATE items:
     • Each has its own distinct quantity and price.
     • They describe clearly different products or services.

2. REFERENCE NUMBERS ARE NOT LINE ITEMS — Strings that look like order/reference codes
   (e.g. INV/2025/60001, SO-01136, PO-123, REF#456, WO-789) written inside the table
   area are internal tracking references, NOT products or services. Do NOT include them
   as separate line items. If a column exists for them (e.g. "Reference", "PO No"),
   include them as a column value on the relevant item row instead.

3. COLUMN HEADERS — Use the EXACT column header text from the invoice as the JSON key.
   Do not rename or merge columns. Preserve every column as its own key.

4. EVERY REAL ROW — Extract every genuine product/service row. Include tax rows,
   service charge rows, and fee rows as separate items if they appear in the table.
   Do NOT include blank/empty rows, totals rows, or sub-header rows as items.

5. MATH VERIFICATION — Before finalising each line item, verify:
     Unit Price × Quantity = Total Amount (allow small rounding differences ≤ 0.05).
   If the numbers do not match, re-read the invoice carefully and correct the value
   that is most likely a misread. Report the corrected values.

6. GRAND TOTAL VERIFICATION — After extracting all line items, sum their totals and
   confirm it reconciles with the invoice's stated grand total (before or after VAT).
   If there is a mismatch, flag it by adding a key "Total_Mismatch": true to the JSON."""

    prompt = (
        "You are an AI specialized in extracting structured data from invoices. "
        "The invoice may be digital, printed, or handwritten. "
        "It may contain text in English, Arabic, or both. "
        "Apply semantic reasoning — do not mechanically copy text. Understand what each "
        "piece of information MEANS before deciding where it belongs.\n\n"
        "Your response must always be a valid JSON object, using EXACTLY these keys:\n"
        "```json\n"
        f"{schema}\n"
        "```\n"
        "Ensure the JSON structure remains consistent and does not wrap data in extra keys like 'Invoice'. "
        "If a value is not present on the invoice, use an empty string.\n\n"
        "DATE RULE (critical): Extract dates EXACTLY as written on the invoice — same format, same order, same digits. "
        "If the invoice shows '23/1' copy '23/1'. If it shows '23/1/2023' copy '23/1/2023'. "
        "Do NOT reorder day/month/year. Do NOT convert formats (e.g. do not turn '23/1/2023' into 'Jan 23, 2023'). "
        "Do NOT invent or assume any part of a date that is not explicitly printed on the invoice "
        "(e.g. if the year is missing, leave it missing — do not guess it). "
        "If no date is found, use an empty string.\n\n"
        "LANGUAGE RULE (critical): Copy every text value EXACTLY as it appears on the invoice — "
        "same script, same language, same characters. "
        "If a word is written in Arabic script, output it in Arabic script (e.g. اشعال عامة). "
        "If it is in English, output it in English. "
        "NEVER transliterate: do not convert Arabic to Latin letters (e.g. do NOT write 'Ash\\'al Amma' for اشعال عامة). "
        "Do not translate. Do not mix scripts within a single field value unless the invoice itself does so.\n\n"
        "GENERAL FIELD RULES:\n"
        "- Invoice Number = the unique identifier for this specific invoice/receipt/transaction. "
        "It may be labelled: Invoice No, رقم الفاتورة, Receipt No, Ref, Cash No, Folio No, Bill No, Doc No, No., or similar. "
        "It may also appear as a value alongside the invoice type label (e.g. 'cash 9290' or 'Tax Invoice 1924') — "
        "in that case the number portion (e.g. '9290', '1924', or the full string 'cash 9290') IS the invoice number. "
        "Do not leave this empty if any reference number or document identifier appears anywhere on the invoice.\n"
        "- Supplier Name = the company/entity that ISSUED the invoice (the seller).\n"
        "- Customer Name = the entity (person OR company) that is BUYING / being BILLED. "
        "Step 1 — find the field explicitly labelled as the customer/buyer: "
        "Customer, اسم العميل, Client, Bill To, Sold To, Billed To, Guest Name, Guest, "
        "Employee, Passenger, Purchaser, Payee, Attn, Cardholder, or any similar buyer label. "
        "Use that field's value — whether it is a company name or a person's name. "
        "Step 2 — CRITICAL exclusions: NEVER use a name that is labelled as the seller/supplier side: "
        "Seller (البائع), Vendor, Sales Rep, Cashier, Issued By, Prepared By, Salesman, "
        "Authorised By, Signed By, or any label on the supplier/issuer side of the invoice. "
        "These people work for the supplier — they are NOT the customer. "
        "Step 3 — if no explicit buyer label exists, use the entity whose VAT/CR number appears "
        "in the customer VAT / buyer registration field.\n"
        "- Total Amount After VAT = the final grand total payable including ALL taxes "
        "and charges (labelled Total / Grand Total / Balance / Amount Due / إجمالي المبلغ). "
        "It must be ≥ Amount Before VAT. Never confuse it with a subtotal or VAT amount alone.\n"
        "- For any custom/additional field not in the list above, apply semantic reasoning: "
        "understand what that field concept MEANS on any invoice format and find the "
        "equivalent value even if the label wording differs.\n\n"
        "Extract the following details from this invoice:\n"
        + "\n".join(bullet_lines)
        + line_items_rules
    )

    # When we have a reliable text layer, hand it to Gemini explicitly. We still
    # also send the image (in process_invoice) so the model can use layout/visual
    # cues — text + image together is the most accurate combination for digital PDFs.
    if source_text:
        prompt += (
            "\n\nThe following is the EXACT machine-extracted text layer from this "
            "invoice (lossless, authoritative for spellings, numbers and VAT/ID strings). "
            "Prefer it for exact characters; use the image only for layout/table structure:\n"
            "<<<INVOICE_TEXT_START>>>\n"
            f"{source_text}\n"
            "<<<INVOICE_TEXT_END>>>"
        )

    # ── Pretty-print the prompt to the terminal so you can inspect what goes to Gemini ──
    width = 70
    print("\n" + "█" * width)
    print(f"█{'  PROMPT SENT TO GEMINI':^{width-2}}█")
    print("█" * width)
    for line in prompt.splitlines():
        # Wrap long lines
        while len(line) > width - 4:
            print(f"  {line[:width-4]}")
            line = "  " + line[width-4:]
        print(f"  {line}")
    print("█" * width)
    print(f"█{'  TOTAL CHARS: ' + str(len(prompt)):^{width-2}}█")
    print("█" * width + "\n")

    return prompt


#--------------------------------------------------------------------------HYBRID SOURCE DETECTION--------------------------------------------------------------------------
def _normal_char_ratio(text):
    """Fraction of characters that are ordinary printable latin/arabic/digit/punct.
    A low ratio means the extracted 'text' is mojibake from a broken font/CID map,
    which is worse than useless -> we should OCR instead."""
    if not text:
        return 0.0
    total = len(text)
    # latin, digits, common punct/space
    # Arabic blocks:
    #   \u0600-\u06FF  Arabic core
    #   \u0750-\u077F  Arabic Supplement
    #   \u08A0-\u08FF  Arabic Extended-A
    #   \uFB50-\uFDFF  Arabic Presentation Forms-A
    #   \uFE70-\uFEFF  Arabic Presentation Forms-B
    normal = len(re.findall(
        r"[A-Za-z0-9\s\.,:\-\/\(\)\#\%\&\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]",
        text
    ))
    return normal / total if total else 0.0


def _page_is_digital(page, cfg=DIGITAL_DETECTION_CONFIG):
    """Decide whether a SINGLE PyMuPDF page has a usable, authoritative text layer.

    Returns (is_digital: bool, text: str, reason: str).

    A page is DIGITAL only when ALL of these hold:
      * It exposes enough real text (char + word count above thresholds).
      * That text is not garbage/encoded (normal-char ratio high enough).
      * It is NOT a scan wrapped in a PDF (a near-full-page image with no text).

    Anything that fails -> treated as scanned/handwritten -> Vision/OCR path.
    This is intentionally biased toward the safe VISION fallback.
    """
    text = page.get_text("text") or ""
    stripped = text.strip()
    words = page.get_text("words")  # list of word boxes
    n_chars = len(stripped)
    n_words = len(words)

    page_area = float(page.rect.width * page.rect.height) or 1.0

    # --- Signal 1: is there a near-full-page image (classic scanned page)? ---
    max_image_cover = 0.0
    try:
        for img in page.get_images(full=True):
            try:
                bbox = page.get_image_bbox(img)
                cover = (bbox.width * bbox.height) / page_area
                max_image_cover = max(max_image_cover, cover)
            except Exception:
                # Some images have no resolvable bbox; ignore for coverage purposes.
                continue
    except Exception:
        max_image_cover = 0.0

    # Hard scan signal: big image AND essentially no text -> definitely OCR.
    if max_image_cover >= cfg["full_page_image_cover"] and n_chars < cfg["min_text_when_large_image"]:
        return False, text, f"full_page_image(cover={max_image_cover:.2f}, chars={n_chars})"

    # --- Signal 2: not enough text to trust ---
    if n_chars < cfg["min_chars_per_page"]:
        return False, text, f"too_little_text(chars={n_chars})"
    if n_words < cfg["min_words_per_page"]:
        return False, text, f"too_few_words(words={n_words})"

    # --- Signal 3: garbage / broken-encoding text layer ---
    ratio = _normal_char_ratio(text)
    if ratio < cfg["min_normal_char_ratio"]:
        return False, text, f"garbage_text(normal_ratio={ratio:.2f})"

    # Passed every gate -> safe to use the text layer.
    return True, text, f"digital(chars={n_chars}, words={n_words}, normal_ratio={ratio:.2f}, img_cover={max_image_cover:.2f})"


def is_source_digital(file_bytes, mime_type, filename="", cfg=DIGITAL_DETECTION_CONFIG):
    """PRODUCTION ENTRY POINT for source-type routing.

    Decides whether a document should go down the cheap/lossless DIGITAL path
    (use the embedded text layer) or the VISION path (rasterise + Gemini Vision OCR).

    Handles every real-world case the pipeline can receive:
      * PNG / JPG / JPEG / TIFF / BMP / WEBP  -> NEVER digital (pixels only) -> Vision.
      * Digital ("born-digital") PDF           -> DIGITAL when text layer is good.
      * Scanned PDF (image wrapped in a PDF)    -> Vision.
      * Handwritten PDF/image                   -> Vision (no/garbage text layer).
      * Mixed/hybrid multi-page PDF             -> per-page text captured; the doc
                                                   is digital only if EVERY page is
                                                   digital, otherwise Vision (safe).
      * Corrupt / unreadable file               -> Vision (fail safe).

    Args:
        file_bytes : raw bytes of the uploaded file.
        mime_type  : the uploaded file's MIME type (e.g. 'application/pdf',
                     'image/png'). Falls back to the filename extension if blank.
        filename   : original filename, used only to recover the type if mime
                     is missing/unreliable.
        cfg        : detection thresholds (see DIGITAL_DETECTION_CONFIG).

    Returns:
        dict with:
          is_digital   : bool   -> True => use text path, False => use Vision path.
          page_texts   : list[str] -> extracted text per page (empty list for images
                                      or when nothing extractable). Index aligns with
                                      PDF page order so callers can pair text with the
                                      rendered image of the same page.
          page_count   : int
          reasons      : list[str] -> per-page human-readable decision reason (for logs).
          source_kind  : str    -> 'image' | 'digital_pdf' | 'scanned_pdf' |
                                    'mixed_pdf' | 'unknown'
    """
    # ---- Resolve the type robustly (mime first, then extension) ----
    mime = (mime_type or "").lower()
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "")

    image_exts = {"png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp", "gif"}
    is_image = mime.startswith("image/") or ext in image_exts
    is_pdf = (mime == "application/pdf") or ext == "pdf"

    # ---- Images can never carry a text layer -> always Vision ----
    if is_image:
        logger.info("is_source_digital: IMAGE (%s) -> VISION path", mime or ext or "unknown")
        return {
            "is_digital": False,
            "page_texts": [],
            "page_count": 1,
            "reasons": ["image_no_text_layer"],
            "source_kind": "image",
        }

    # ---- Anything we cannot identify as a PDF: fail safe to Vision ----
    if not is_pdf:
        logger.warning("is_source_digital: UNKNOWN type (mime=%s ext=%s) -> VISION path", mime, ext)
        return {
            "is_digital": False,
            "page_texts": [],
            "page_count": 0,
            "reasons": ["unknown_type_fallback_vision"],
            "source_kind": "unknown",
        }

    # ---- PDF: open and inspect every page ----
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        logger.error("is_source_digital: PDF open failed (%s) -> VISION path", e)
        return {
            "is_digital": False,
            "page_texts": [],
            "page_count": 0,
            "reasons": [f"pdf_open_error:{e}"],
            "source_kind": "unknown",
        }

    page_texts, reasons, per_page_digital = [], [], []
    try:
        for i, page in enumerate(doc):
            ok, text, reason = _page_is_digital(page, cfg)
            page_texts.append(text or "")
            per_page_digital.append(ok)
            reasons.append(f"p{i + 1}:{reason}")
        page_count = len(per_page_digital)
    finally:
        doc.close()

    if page_count == 0:
        return {
            "is_digital": False, "page_texts": [], "page_count": 0,
            "reasons": ["empty_pdf"], "source_kind": "unknown",
        }

    n_digital = sum(per_page_digital)
    # Whole document is digital ONLY if every page is digital. A single scanned /
    # handwritten page forces the safe Vision route for the document so we never
    # silently drop the un-extractable page's data.
    all_digital = (n_digital == page_count)

    if all_digital:
        source_kind = "digital_pdf"
    elif n_digital == 0:
        source_kind = "scanned_pdf"
    else:
        source_kind = "mixed_pdf"

    logger.info(
        "is_source_digital: PDF %s pages=%d digital_pages=%d -> %s | %s",
        source_kind, page_count, n_digital,
        "DIGITAL" if all_digital else "VISION", "; ".join(reasons),
    )

    return {
        "is_digital": all_digital,
        "page_texts": page_texts,
        "page_count": page_count,
        "reasons": reasons,
        "source_kind": source_kind,
    }


#--------------------------------------------------------------------------IMAGE DATA EXTRACTION--------------------------------------------------------------------------
import os
from datetime import datetime

QR_DEBUG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qrcode_out.txt")

def _qr_log(text):
    with open(QR_DEBUG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def extract_qr_code(image_data):
    """Detects and extracts QR code content from the image.
    Tries multiple preprocessed versions so low-res / compressed images still work.
    """
    nparr = np.frombuffer(image_data, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original is None:
        _qr_log("STATUS      : NOT FOUND — image could not be decoded by OpenCV.")
        return None

    _qr_log("")
    _qr_log("=" * 60)
    _qr_log(f"Scanned at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _qr_log(f"Image size : {original.shape[1]}x{original.shape[0]} px")

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Pre-built reusable intermediates
    up2  = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    up3  = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    up4  = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Sharpen kernel — helps blurry / low-res QR codes
    sharpen_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened   = cv2.filter2D(gray, -1, sharpen_k)
    sharp_up2   = cv2.filter2D(up2,  -1, sharpen_k)
    sharp_up3   = cv2.filter2D(up3,  -1, sharpen_k)

    # CLAHE — improves contrast on faded/washed-out prints
    clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray  = clahe.apply(gray)
    clahe_up2   = clahe.apply(up2)

    def otsu(img):
        return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def adaptive(img, block=11, c=2):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block, c)

    # Build a list of image variants to try, from cheapest to most aggressive
    variants = [
        ("original colour",        original),
        ("grayscale",              gray),
        ("upscaled 2x",            up2),
        ("upscaled 3x",            up3),
        ("upscaled 4x",            up4),
        ("otsu",                   otsu(gray)),
        ("adaptive 11",            adaptive(gray, 11, 2)),
        ("adaptive 15",            adaptive(gray, 15, 4)),
        ("sharpened",              sharpened),
        ("sharpened+otsu",         otsu(sharpened)),
        ("sharp+up2",              sharp_up2),
        ("sharp+up2+otsu",         otsu(sharp_up2)),
        ("sharp+up3",              sharp_up3),
        ("sharp+up3+otsu",         otsu(sharp_up3)),
        ("clahe",                  clahe_gray),
        ("clahe+otsu",             otsu(clahe_gray)),
        ("clahe+up2",              clahe_up2),
        ("clahe+up2+otsu",         otsu(clahe_up2)),
        ("up2+adaptive",           adaptive(up2, 11, 2)),
        ("up3+otsu",               otsu(up3)),
        ("up3+adaptive",           adaptive(up3, 15, 4)),
        ("up4+otsu",               otsu(up4)),
    ]

    for name, img in variants:
        qr_codes = decode(img)
        _qr_log(f"  Tried [{name}] → {'FOUND' if qr_codes else 'not found'}")
        if qr_codes:
            raw = qr_codes[0].data.decode("utf-8")
            _qr_log(f"STATUS      : FOUND (via {name})")
            _qr_log(f"RAW BASE64  : {raw}")
            return raw

    _qr_log("STATUS      : NOT FOUND — all preprocessing variants failed.")
    return None

def _scan_tlv_string(qr_bytes, target_tag):
    """Scan the byte stream for a ZATCA tag and return its value as a clean string.
    Scanning (rather than strict sequential parsing) survives QRs with a
    corrupt/garbled field that would otherwise knock the parser out of alignment.
    """
    n = len(qr_bytes)
    i = 0
    while i < n - 1:
        tag = qr_bytes[i]
        length = qr_bytes[i + 1]
        if tag == target_tag and 0 < length <= 60 and i + 2 + length <= n:
            chunk = qr_bytes[i + 2 : i + 2 + length]
            try:
                return chunk.decode("utf-8")
            except UnicodeDecodeError:
                pass
        i += 1
    return None


def _scan_tlv_amount(qr_bytes, target_tag):
    """Scan for a tag whose value is a plain decimal number (totals / VAT amount).
    The strict numeric check makes a false match extremely unlikely."""
    n = len(qr_bytes)
    i = 0
    while i < n - 1:
        tag = qr_bytes[i]
        length = qr_bytes[i + 1]
        if tag == target_tag and 0 < length <= 20 and i + 2 + length <= n:
            chunk = qr_bytes[i + 2 : i + 2 + length]
            try:
                s = chunk.decode("ascii")
                if re.fullmatch(r"\d+(\.\d+)?", s):
                    return float(s)
            except (UnicodeDecodeError, ValueError):
                pass
        i += 1
    return None


def decode_tlv_qr(qr_string):
    """
    Decodes the extracted QR code data (Base64-encoded TLV format)
    used in E-Invoice QR Reader KSA. Robust against malformed timestamp fields.
    """
    try:
        qr_bytes = base64.b64decode(qr_string)

        # tag 1 = seller name, 2 = VAT, 3 = timestamp, 4 = total (with VAT), 5 = VAT amount
        supplier_name = _scan_tlv_string(qr_bytes, 1)
        supplier_vat  = _scan_tlv_string(qr_bytes, 2)
        invoice_date  = _scan_tlv_string(qr_bytes, 3)
        total_amount  = _scan_tlv_amount(qr_bytes, 4)
        vat_amount    = _scan_tlv_amount(qr_bytes, 5)

        # Strip any leftover non-printable bytes from string fields
        def clean(s):
            if s is None:
                return None
            return re.sub(r"[^\x20-\x7E؀-ۿ]", "", s) or None

        result = {
            "Supplier_Name":          clean(supplier_name),
            "Supplier_VAT":           clean(supplier_vat),
            "Invoice_Date":           clean(invoice_date),
            "Total_Amount_After_VAT": total_amount,
            "VAT_Amount":             vat_amount,
        }

        _qr_log("DECODED TLV :")
        _qr_log(f"  Supplier Name  : {result['Supplier_Name']}")
        _qr_log(f"  Supplier VAT   : {result['Supplier_VAT']}")
        _qr_log(f"  Invoice Date   : {result['Invoice_Date']}")
        _qr_log(f"  Total Amt+VAT  : {result['Total_Amount_After_VAT']}")
        _qr_log(f"  VAT Amount     : {result['VAT_Amount']}")

        return result

    except Exception as e:
        _qr_log(f"DECODE ERROR: {e}")
        return {"Error": str(e)}

def extract_images_from_pdf(pdf_data, dpi=RASTER_DPI):
    """Extracts a rendered PNG image for each page of a PDF file.

    Rendering at an explicit DPI (instead of the default ~96) materially improves
    Gemini Vision OCR accuracy and QR detection on scanned/low-res documents.
    """
    images = []
    pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
    try:
        # 72 is the PDF base DPI; scale the matrix to reach the requested DPI.
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_num in range(len(pdf_document)):
            pix = pdf_document[page_num].get_pixmap(matrix=matrix)
            image_data = pix.tobytes("png")  # Convert to PNG byte format
            images.append(image_data)
    finally:
        pdf_document.close()

    return images

def merge_images_vertically(image_list):
    """Merges multiple invoice images vertically into a single image."""

    # Convert byte images to PIL Image objects
    images = [Image.open(io.BytesIO(img)) for img in image_list]

    # Find the max width and total height
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)

    # Create a blank white canvas
    merged_image = Image.new("RGB", (max_width, total_height), "white")

    # Paste images on top of each other
    y_offset = 0
    for img in images:
        # Resize image to match the widest one while keeping aspect ratio
        if img.width < max_width:
            img = img.resize((max_width, int(img.height * (max_width / img.width))))

        merged_image.paste(img, (0, y_offset))
        y_offset += img.height  # Move offset for next image

    # Convert back to bytes
    img_byte_array = io.BytesIO()
    merged_image.save(img_byte_array, format="PNG")
    return img_byte_array.getvalue()


#-----------------------------------------------------------------------INVOICE PROCESSING (HYBRID)-----------------------------------------------------------------------
def _gemini_extract(prompt, image_data=None):
    """Single Gemini call. Sends image+prompt when an image is given, else text-only.
    Returns the parsed dict, or raises on unrecoverable JSON errors."""
    if image_data is not None:
        image = Image.open(io.BytesIO(image_data))
        response = gemini_model.generate_content([prompt, image])
    else:
        response = gemini_model.generate_content(prompt)

    response_text = (response.text or "").strip()

    # Pull JSON out of a ```json ... ``` fence if present, else use raw text.
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        cleaned_json = match.group(1).strip()
    else:
        # Fall back to the first {...} block to survive stray preamble.
        brace = re.search(r"(\{.*\})", response_text, re.DOTALL)
        cleaned_json = brace.group(1).strip() if brace else response_text.strip()

    return json.loads(cleaned_json)


def process_invoice(image_data, selected_fields=None, custom_fields=None, source_text=None):
    """Extract structured data from a single invoice PAGE using Gemini 2.5 Flash.

    selected_fields -> list of PREDEFINED_FIELDS dicts the user chose to extract.
                       Defaults to ALL predefined fields (= original behaviour).
    custom_fields   -> list of extra field names the user typed in the UI.
    source_text     -> OPTIONAL extracted text layer for THIS page (digital path).
                       When present, Gemini receives text + image together, which
                       is the most accurate combination for born-digital PDFs.

    image_data is ALWAYS supplied because:
      * QR/TLV detection (ZATCA) reads the rendered image, not the text layer, so a
        digital PDF still needs its image for the QR cross-check.
      * Sending the image alongside the authoritative text gives Gemini the layout
        cues it needs to keep line items aligned.
    """
    if selected_fields is None:
        selected_fields = PREDEFINED_FIELDS
    if custom_fields is None:
        custom_fields = []

    image = Image.open(io.BytesIO(image_data))

    # Build the prompt dynamically; inject the text layer when we trust it.
    prompt = build_invoice_prompt(selected_fields, custom_fields, source_text=source_text)

    response = gemini_model.generate_content([prompt, image])

    # Extract and clean response
    response_text = response.text.strip()
    # Remove the backticks and "json" label
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        cleaned_json = match.group(1).strip()
    else:
        brace = re.search(r"(\{.*\})", response_text, re.DOTALL)
        cleaned_json = brace.group(1).strip() if brace else response_text.strip()

    # Convert JSON string to dictionary
    invoice_data = json.loads(cleaned_json)

    def safe_float(value, default=0.0):
        """Convert value to float, handling None or invalid values."""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    # ✅ Build the stored record dynamically based on what the user selected.
    new_invoice_data = {}
    selected_keys = {f["key"] for f in selected_fields}

    for field in selected_fields:
        if field["type"] == "line_items":
            # handled separately below
            continue
        default = "0.00" if field["type"] == "float" else (False if field["type"] == "bool" else "Unknown")
        new_invoice_data[field["key"]] = find_value(invoice_data, field["aliases"], default)

    # ✅ Line Items — only if the user asked for them.
    # Keys are kept exactly as Gemini returns them (= the invoice's actual column headers).
    if "Line_Items" in selected_keys:
        line_items = find_value(invoice_data, ["Line Items", "Line_Items", "LineItems", "Line Item", "Line_Item"], [])
        new_invoice_data["Line_Items"] = []
        if isinstance(line_items, list):
            new_invoice_data["Line_Items"] = [
                {k: v for k, v in item.items()}
                for item in line_items if isinstance(item, dict)
            ]

    # ✅ Custom user-defined fields — stored under their own label as the column name
    for cf in custom_fields:
        cf = cf.strip()
        if cf:
            new_invoice_data[cf] = find_value(invoice_data, [cf], "")

    # QR / ZATCA TLV cross-check — ALWAYS runs against the rendered image, for both
    # digital and scanned paths, because the QR payload is authoritative for VAT/total.
    qr_code_string = extract_qr_code(image_data)
    if qr_code_string:
        qr_data = decode_tlv_qr(qr_code_string)
    else:
        qr_data = None

    # QR_Code_Present is set from our own OpenCV scan — NOT from Gemini's visual guess.
    # Gemini can hallucinate this field; pyzbar is the ground truth.
    new_invoice_data['QR_Code_Present'] = qr_code_string is not None

    if qr_data and "Error" not in qr_data:
        new_invoice_data['QR_Code_Valid'] = True
        for key, qr_value in qr_data.items():
            if key in new_invoice_data and qr_value:
                # Supplier_Name: Arabic vs English causes false mismatches
                # Invoice_Date: QR stores a binary timestamp, Gemini reads the printed date correctly
                if key in ("Supplier_Name", "Invoice_Date", "QR_Code_Present"):
                    continue
                if new_invoice_data[key] != qr_value:
                    logger.info("QR override %s: %s -> %s", key, new_invoice_data[key], qr_value)
                    new_invoice_data[key] = qr_value
    else:
        new_invoice_data['QR_Code_Valid'] = False

    return new_invoice_data  # Return consistent structured data


def process_document(file_bytes, mime_type, filename, selected_fields, custom_fields, split_invoices):
    """ORCHESTRATOR: routes a single uploaded file through the hybrid pipeline and
    returns a LIST of extracted invoice records (one file may hold several invoices).

    Routing:
      1. is_source_digital() inspects the file.
      2. We ALWAYS render page image(s) (needed for QR + layout cues).
      3. DIGITAL pages additionally pass their lossless text layer to Gemini.
      4. Images / scanned / handwritten / mixed -> Vision-only (source_text=None).

    The 'split_invoices' flag preserves your existing PDF semantics:
      * False -> the PDF is ONE invoice spanning multiple pages (images merged,
                 text layers concatenated).
      * True  -> each PDF page is a SEPARATE single-page invoice.
    """
    detection = is_source_digital(file_bytes, mime_type, filename)
    is_pdf = detection["source_kind"] in ("digital_pdf", "scanned_pdf", "mixed_pdf") or \
             (mime_type == "application/pdf")

    n_pages   = detection.get("page_count", 0)
    reasons   = detection.get("reasons", [])
    is_dig    = detection["is_digital"]
    kind      = detection["source_kind"]

    print(f"\n{'='*60}")
    print(f"  FILE        : {filename}")
    print(f"  MIME TYPE   : {mime_type}")
    print(f"  PAGES       : {n_pages}")
    print(f"  source_kind : {kind}")
    print(f"{'─'*60}")
    # Per-page breakdown
    for r in reasons:
        # r is like "p1:digital(chars=842, words=97, ...)" or "p1:garbage_text(...)"
        verdict = "✅ DIGITAL" if "digital(" in r else "❌ NOT DIGITAL"
        print(f"  {verdict}  →  {r}")
    print(f"{'─'*60}")
    # Final verdict with explanation
    if kind == "image":
        print(f"  VERDICT : ❌ NOT DIGITAL  — plain image file, no text layer possible")
        print(f"  PATH    : Vision only (Gemini reads pixels)")
    elif is_dig:
        print(f"  VERDICT : ✅ DIGITAL  — all {n_pages} page(s) have a clean text layer")
        print(f"  PATH    : Text layer sent to Gemini + Vision for layout")
    elif kind == "scanned_pdf":
        print(f"  VERDICT : ❌ NOT DIGITAL  — PDF is a scanned image, no usable text layer")
        print(f"  PATH    : Vision only (Gemini reads pixels)")
    elif kind == "mixed_pdf":
        n_fail = sum(1 for r in reasons if "digital(" not in r)
        print(f"  VERDICT : ❌ NOT DIGITAL  — {n_fail}/{n_pages} page(s) failed the text check")
        print(f"  PATH    : Vision only (one bad page forces full Vision fallback)")
    else:
        print(f"  VERDICT : ❌ NOT DIGITAL  — {kind}")
        print(f"  PATH    : Vision only")
    print(f"{'='*60}")

    records = []

    if not is_pdf:
        # ---- Plain image upload: always Vision, single record ----
        rec = process_invoice(file_bytes, selected_fields, custom_fields, source_text=None)
        if rec:
            rec["_source_kind"] = detection["source_kind"]
            rec["_extraction_path"] = "vision"
            records.append(rec)
        return records

    # ---- PDF: render every page once ----
    page_images = extract_images_from_pdf(file_bytes)
    page_texts = detection["page_texts"]
    use_text = detection["is_digital"]  # whole-doc digital gate

    # Helper to safely align text with a page index.
    def text_for(idx):
        if use_text and idx < len(page_texts):
            t = (page_texts[idx] or "").strip()
            return t or None
        return None

    if not split_invoices:
        # ONE multi-page invoice: merge images, concatenate the trusted text layers.
        merged_image = merge_images_vertically(page_images) if len(page_images) > 1 else page_images[0]
        merged_text = None
        if use_text:
            joined = "\n".join((page_texts[i] or "") for i in range(len(page_texts))).strip()
            merged_text = joined or None
        rec = process_invoice(merged_image, selected_fields, custom_fields, source_text=merged_text)
        if rec:
            rec["_source_kind"] = detection["source_kind"]
            rec["_extraction_path"] = "digital_text+vision" if merged_text else "vision"
            records.append(rec)
    else:
        # MULTIPLE single-page invoices: one record per page.
        for idx, img in enumerate(page_images):
            rec = process_invoice(img, selected_fields, custom_fields, source_text=text_for(idx))
            if rec:
                rec["_source_kind"] = detection["source_kind"]
                rec["_extraction_path"] = "digital_text+vision" if text_for(idx) else "vision"
                records.append(rec)

    return records


#--------------------------------------------------------------------------CHATBOT INTEGRATION--------------------------------------------------------------------------
from datetime import datetime
from dateutil import parser


def get_filtered_invoices(query):
    """Parse query text to extract filters, then query MongoDB."""
    filters = {}

    # --- Extract basic filters ---
    month = extract_month(query)
    year = extract_year(query)
    quarter, q_year = extract_quarter(query)

    if quarter and q_year:
        start_month = (quarter - 1) * 3 + 1
        filters["Invoice_Date"] = {
            "$gte": datetime(q_year, start_month, 1),
            "$lt": datetime(q_year, start_month + 3, 1)
        }

    elif month and year:
        filters["Invoice_Date"] = {
            "$gte": datetime(year, month, 1),
            "$lt": datetime(year, month + 1 if month < 12 else 1, 1)
        }

    elif year:
        filters["Invoice_Date"] = {
            "$gte": datetime(year, 1, 1),
            "$lt": datetime(year + 1, 1, 1)
        }

    if "vat valid" in query.lower():
        filters["QR_Code_Valid"] = True

    if "invalid vat" in query.lower():
        filters["QR_Code_Valid"] = False

    # --- Query MongoDB ---
    results = list(invoice_collection.find(filters))
    return results


def extract_month(query):
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    for name, number in months.items():
        if name in query.lower():
            return number
    return None

def extract_year(query):
    now = datetime.now()
    if "last year" in query.lower() or "previous year" in query.lower():
        return now.year - 1
    elif "this year" in query.lower() or "current year" in query.lower():
        return now.year
    match = re.search(r"(20\d{2})", query)
    return int(match.group(1)) if match else None

def extract_quarter(query):
    match = re.search(r"q([1-4])[\s\-]?(20\d{2})?", query.lower())
    if match:
        quarter = int(match.group(1))
        year = int(match.group(2)) if match.group(2) else datetime.now().year
        return quarter, year

    return None, None


def llm_output(invoices, query):
    data = ""
    for invoice in invoices:
        line_items = invoice.get('Line_Items', [])
        if not isinstance(line_items, list):
            line_items = []

        try:
            item_names = [item.get('Item_Name', 'Unknown') for item in line_items]
        except Exception as e:
            item_names = []
            print(f"Error while iterating line_items: {e}")

        data += f"""
            - Invoice Number: {invoice.get('Invoice_Number')}
              Date: {invoice.get('Invoice_Date')}
              Supplier Name: {invoice.get('Supplier_Name')}
              Supplier VAT: {invoice.get('Supplier_VAT')}
              Customer Name: {invoice.get('Customer_Name')}
              Customer VAT: {invoice.get('Customer_VAT')}
              Amount Before VAT: {invoice.get('Amount_Before_VAT')}
              VAT Amount: {invoice.get('VAT_Amount')}
              Total Amount After VAT: {invoice.get('Total_Amount_After_VAT')}
              QR Code Present: {invoice.get('QR_Code_Present')}
              Items: {item_names}\n
        """

    prompt = (
        "You are an invoice data assistant that returns answers to the users queries "
        "in a structured manner, you are thorough and accurate.\n\n"
        f"User question: {query}\n"
        f"Here is the invoice data:\n{data}\n"
        "Now provide a concise summary."
    )

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"LLM Error: {e}"


def normalize_date(date_str):
    try:
        parsed = parser.parse(str(date_str), dayfirst=True, fuzzy=True)
        return parsed.strftime("%d/%m/%y")
    except Exception as e:
        print(f"[Date Normalization Error]: {e} — Input: {date_str}")
        return date_str


def handle_query(query, project):
    filters = get_filtered_invoices(query)

    if not isinstance(filters, list):
        return "❌ Unexpected data structure."

    if not filters:
        filters = list(invoice_collection.find({"Project": project}))

    return llm_output(filters, query)


#--------------------------------------------------------------------------STREAMLIT APP--------------------------------------------------------------------------
st.title("AI Petty Cash Manager")

# Sidebar: Add New Project
st.sidebar.image('JFF-LOGO-White-removebg.png', width=100)
st.sidebar.header("Project Management")
project_name = st.sidebar.text_input("Enter New Project Name")

if st.sidebar.button("Add Project"):
    if project_name:
        if project_name and not projects_coll.find_one({"project_name": project_name}):
            projects_coll.insert_one({"project_name": project_name})
            st.sidebar.success(f"Project '{project_name}' added!")
        else:
            st.sidebar.warning("Project already exists!")
    else:
        st.sidebar.error("Project name cannot be empty.")

# Select Project Dropdown
selected_project = st.sidebar.selectbox("Select a Project", [p['project_name'] for p in projects_coll.find()])

# Navigation Options
page_selection = st.sidebar.radio("Navigation", ["Project Overview", "Analytics"])


if selected_project not in st.session_state.projects:
    st.session_state.projects[selected_project] = []

if f"{selected_project}_supplier_vat_missing_count" not in st.session_state:
    st.session_state[f"{selected_project}_supplier_vat_missing_count"] = 0

if f"{selected_project}_missing_data_records" not in st.session_state:
    st.session_state[f"{selected_project}_missing_data_records"] = []

if selected_project:
    st.subheader(f"Project: {selected_project}")

#--------------------------------------------------------------------------PAGE: PROJECT OVERVIEW--------------------------------------------------------------------------
    if page_selection == "Project Overview":

        #------------------------------------------------------------------FIELD SELECTION (DYNAMIC PROMPT)------------------------------------------------------------------
        # 1) Additional custom fields the user wants on top of the predefined ones
        with st.expander("⚙️ Configure fields to extract", expanded=True):
            st.markdown("#### ➕ Additional fields")
            st.caption("Add any field that isn't in the predefined list (e.g. `VAT %`, `Municipality Tax`, `Room No`).")

            if "custom_fields" not in st.session_state:
                st.session_state.custom_fields = [""]

            # Render an input box for each custom field slot
            for i in range(len(st.session_state.custom_fields)):
                st.session_state.custom_fields[i] = st.text_input(
                    f"Additional field {i + 1}",
                    value=st.session_state.custom_fields[i],
                    key=f"custom_field_{i}",
                    placeholder="e.g. VAT %",
                )

            col_add, col_remove = st.columns(2)
            with col_add:
                if st.button("➕ Add another field", use_container_width=True):
                    st.session_state.custom_fields.append("")
                    st.rerun()
            with col_remove:
                if st.button("➖ Remove last field", use_container_width=True):
                    if len(st.session_state.custom_fields) > 1:
                        st.session_state.custom_fields.pop()
                        st.rerun()

            st.divider()

            # 2) Checklist of predefined fields
            st.markdown("#### ✅ Predefined fields")
            st.caption("Tick the fields you want extracted from each invoice.")
            selected_fields = []
            check_cols = st.columns(3)
            for idx, field in enumerate(PREDEFINED_FIELDS):
                with check_cols[idx % 3]:
                    if st.checkbox(field["label"], value=True, key=f"chk_{field['key']}"):
                        selected_fields.append(field)

        # Clean up custom fields (drop blanks / duplicates)
        custom_fields = []
        for cf in st.session_state.custom_fields:
            cf = cf.strip()
            if cf and cf not in custom_fields:
                custom_fields.append(cf)

        # Upload multiple invoices
        uploaded_files = st.file_uploader("Upload Invoices (PNG, JPG, PDF)", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)
        pdf_type = st.radio("Is your PDF file:", ["One Invoice (Multiple Pages)", "Multiple Single-Page Invoices"])
        if st.button("Process Invoices"):
            if not selected_fields and not custom_fields:
                st.error("Please select at least one field to extract.")
            elif uploaded_files:
                existing_invoices = {
                    inv.get("Invoice_Number")
                    for inv in st.session_state.projects.get(selected_project, [])
                }

                new_invoices = []
                repeated_invoices = []
                failed_files = []

                split_invoices = (pdf_type == "Multiple Single-Page Invoices")

                progress = st.progress(0.0)
                status = st.empty()

                for f_idx, uploaded_file in enumerate(uploaded_files):
                    status.info(f"Processing {uploaded_file.name} …")
                    try:
                        file_data = uploaded_file.read()

                        # ---- HYBRID ROUTING happens inside process_document ----
                        extracted_records = process_document(
                            file_bytes=file_data,
                            mime_type=uploaded_file.type,
                            filename=uploaded_file.name,
                            selected_fields=selected_fields,
                            custom_fields=custom_fields,
                            split_invoices=split_invoices,
                        )
                    except Exception as e:
                        logger.exception("Failed to process %s", uploaded_file.name)
                        failed_files.append(f"{uploaded_file.name} ({e})")
                        extracted_records = []

                    for invoice_data in extracted_records:
                        if not invoice_data:
                            continue

                        invoice_number = invoice_data.get("Invoice_Number")

                        if invoice_number in existing_invoices:
                            repeated_invoices.append(invoice_number)
                            continue

                        invoice_data["File Name"] = uploaded_file.name
                        invoice_data["Project"] = selected_project

                        new_invoices.append(invoice_data)
                        existing_invoices.add(invoice_number)

                    progress.progress((f_idx + 1) / len(uploaded_files))

                status.empty()
                progress.empty()

                # Save to session only
                st.session_state.projects[selected_project].extend(new_invoices)

                if new_invoices:
                    st.success(f"Processed {len(new_invoices)} invoice(s) successfully!")

                if repeated_invoices:
                    st.warning(
                        f"Skipped {len(repeated_invoices)} duplicate invoice(s): "
                        f"{', '.join(str(x) for x in repeated_invoices)}"
                    )

                if failed_files:
                    st.error(
                        f"Failed to process {len(failed_files)} file(s): "
                        f"{', '.join(failed_files)}"
                    )

        # Display Invoices in a Table
        if selected_project:
            missing_data_records = []
            invoices = st.session_state.projects.get(selected_project, [])
            if invoices:
                # Convert to DataFrame
                df = pd.DataFrame(invoices)

                # Drop MongoDB `_id` field and internal/audit columns + line items
                df = df.drop(columns=["_id", "Line_Items", "_source_kind", "_extraction_path"], errors="ignore")

                # Clean column names
                df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")

                # Convert amounts to numeric
                for col in ["Total_Amount", "VAT_Amount", "Amount_Before_VAT", "Total_Amount_After_VAT"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

                # Display invoices table
                st.dataframe(df)
                required_fields = ["Invoice_Number", "Invoice_Date", "Supplier_Name", "Supplier_VAT",
                                   "Customer_Name", "Customer_VAT", "Amount_Before_VAT", "VAT_Amount", "Total_Amount"]

                # Display Line Items
                for invoice in invoices:
                    invoice_number = invoice.get("Invoice_Number", "Unknown")
                    missing_fields = [field for field in required_fields if not invoice.get(field)]
                    if missing_fields:
                        missing_data_records.append({
                            "Invoice_Number": invoice.get("Invoice_Number", "Unknown"),
                            "Missing_Fields": ", ".join(missing_fields)
                        })
                    line_items = invoice.get("Line_Items", [])

                    if line_items:
                        with st.expander(f"📌 View Line Items for Invoice: {invoice_number}"):
                            line_items_df = pd.DataFrame(line_items)

                            # Clean column names
                            line_items_df.columns = line_items_df.columns.str.replace(" ", "_").str.replace("-", "_")

                            # Convert numerical values
                            numeric_columns = ["Quantity", "Unit_Price", "Total_Price"]
                            for col in numeric_columns:
                                if col in line_items_df.columns:
                                    line_items_df[col] = pd.to_numeric(line_items_df[col], errors="coerce").fillna(0)

                            st.dataframe(line_items_df, use_container_width=True)

            else:
                st.warning("No invoices found for this project.")


#--------------------------------------------------------------------------CHATBOT FUNCTIONALITY--------------------------------------------------------------------------
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            st.markdown("""
            <style>
            .chat-container {
                padding: 10px;
            }
            .user-bubble {
                background-color: #262730;
                color: white;
                padding: 10px 15px;
                margin-bottom: 5px;
                border-radius: 15px;
                max-width: 80%;
                align-self: flex-end;
                margin-left: auto;
            }
            .bot-bubble {
                background-color: #262730;
                color: white;
                padding: 10px 15px;
                margin-bottom: 5px;
                border-radius: 15px;
                max-width: 80%;
                align-self: flex-start;
                margin-right: auto;
            }
            </style>
            """, unsafe_allow_html=True)

            # Input box first
            st.markdown("### 💬 Ask a question about your invoices")
            query = st.text_input("Type your question:", key="user_query")

            left_col, mid_col, right_col = st.columns([2, 2, 2])
            with left_col:
                if st.button("Clear Chat History", use_container_width=True):
                    st.session_state.chat_history = []
            with right_col:
                submit_clicked = st.button("Submit", use_container_width=True)

            if submit_clicked and query.strip():
                with st.spinner("Processing your query..."):
                    try:
                        response = handle_query(query, selected_project)
                        st.session_state.chat_history.append((query, response))
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            if st.session_state.get("chat_history"):
                st.markdown("### Chat History")
                for user_msg, bot_msg in st.session_state.chat_history:
                    st.markdown(f'<div class="chat-container"><div class="user-bubble">{user_msg}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-container"><div class="bot-bubble">{bot_msg}</div></div>', unsafe_allow_html=True)


#--------------------------------------------------------------------------PAGE: ANALYTICS--------------------------------------------------------------------------
    elif page_selection == "Analytics":
        st.title("📊 Project Analytics")
        # Embed Zoho Analytics Dashboard
        zoho_dashboard_url = "https://analytics.zoho.com/open-view/3032881000000004219"

        # Using Streamlit's iframe component
        st.components.v1.iframe(zoho_dashboard_url, width=800, height=600)
        invoices = st.session_state.projects.get(selected_project, [])
        total_invoices = len(invoices)  # Total invoices in the project

        if total_invoices > 0:
            # Count invoices with and without Supplier VAT
            invoices_with_supplier_vat = sum(1 for inv in invoices if inv.get("Supplier_VAT"))
            supplier_vat_missing_count = total_invoices - invoices_with_supplier_vat

            vat_data = {
                "Category": ["Has Supplier VAT", "Missing Supplier VAT"],
                "Count": [invoices_with_supplier_vat, supplier_vat_missing_count]
            }
            vat_df = pd.DataFrame(vat_data)

            # Create a Donut Chart
            fig = px.pie(vat_df, names="Category", values="Count", hole=0.4,
                         title="Invoices with vs. without Supplier VAT",
                         color_discrete_sequence=["#1f77b4", "#ff7f0e"])  # Blue & Orange

            # Display Donut Chart in Streamlit
            st.plotly_chart(fig)

            # Count invoices with a QR code, handling both key variations and ensuring boolean values
            qr_code_present_count = sum(
                bool(inv.get("QR_Code_Present") in [True, "True", 1]) for inv in invoices
            )
            qr_code_missing_count = total_invoices - qr_code_present_count

            qr_data = {
                "QR Code Status": ["With QR Code", "Without QR Code"],
                "Count": [qr_code_present_count, qr_code_missing_count]
            }
            qr_df = pd.DataFrame(qr_data)

            # Create a Bar Chart
            fig_qr = px.bar(qr_df, x="QR Code Status", y="Count",
                            title="Invoices with vs. without QR Code",
                            color="QR Code Status",
                            color_discrete_sequence=["#2ca02c", "#d62728"],  # Green & Red
                            text="Count")

            # Display Bar Chart in Streamlit
            st.plotly_chart(fig_qr)
        else:
            st.warning("No invoices found for this project.")