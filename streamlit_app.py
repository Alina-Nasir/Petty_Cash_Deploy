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


#--------------------------------------------------------------------------DYNAMIC FIELD CONFIGURATION--------------------------------------------------------------------------
# Each predefined field has:
#   label   -> how it is shown to the user / asked from the LLM
#   key     -> the canonical snake_case key we store it under (keeps old code working)
#   type    -> string / float / bool / line_items (controls the JSON schema hint)
#   aliases -> spellings the LLM might return, used when reading the value back
PREDEFINED_FIELDS = [
    {"label": "Invoice Number", "key": "Invoice_Number", "type": "string",
     "aliases": ["Invoice Number", "Invoice_Number", "InvoiceNumber", "Invoice No", "Invoice No."]},
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


def build_invoice_prompt(selected_fields, custom_fields):
    """Build the Gemini prompt dynamically from the user's chosen fields.

    selected_fields -> list of PREDEFINED_FIELDS dicts the user ticked
    custom_fields   -> list of extra field-name strings the user typed
    """
    schema_lines = []
    bullet_lines = []

    for field in selected_fields:
        if field["type"] == "line_items":
            schema_lines.append(
                '  "Line Items": [\n'
                '    {\n'
                '      "Item Name": "<string>",\n'
                '      "Item Description": "<string>",\n'
                '      "Quantity": <int>,\n'
                '      "Unit Price": <float>,\n'
                '      "Total Price": <float>\n'
                '    }\n'
                '  ]'
            )
            bullet_lines.append("- Line Items (Item Name, Item Description, Quantity, Unit Price, Total Price)")
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

    line_items_instruction = (
        "\n\nIMPORTANT for Line Items: "
        "You MUST extract EVERY SINGLE row from the invoice table — do not skip any. "
        "Count the rows yourself before responding and make sure the array length matches. "
        "Include tax rows, service charge rows, and fee rows as separate line items too."
        if has_line_items else ""
    )

    prompt = (
        "You are an AI specialized in extracting structured data from invoices. "
        "The invoice may contain text in English or Arabic, or both. "
        "The supplier name is the company that issued the invoice. "
        "Your response must always be a valid JSON object, using EXACTLY these keys, formatted as follows:\n"
        "```json\n"
        f"{schema}\n"
        "```\n"
        "Ensure the JSON structure remains consistent and does not wrap data in extra keys like 'Invoice'. "
        "If a value is not present on the invoice, use an empty string.\n"
        "FIELD RULES:\n"
        "- Customer Name = the INDIVIDUAL PERSON associated with the transaction. "
        "Look for fields labelled: Guest Name, Guest, Employee, Passenger, Purchaser, Payee, Attn, Contact, Cardholder, Client, or any similar label that refers to a specific human being. "
        "If a company/organisation name also appears (e.g. Bill To, Company, Employer), ignore it for this field — that is NOT the Customer Name. "
        "If no individual person name exists anywhere on the invoice, only then fall back to the company/organisation name.\n"
        "- Total Amount After VAT = the final grand total payable INCLUDING all taxes and charges "
        "(labelled Total / Grand Total / Balance / Amount Due). It must be greater than or equal to 'Amount Before VAT'. "
        "Do not confuse it with a subtotal, a single line item, or the VAT amount.\n"
        "Extract the following details from this invoice:\n"
        + "\n".join(bullet_lines)
        + line_items_instruction
    )
    print(prompt)
    return prompt


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

    _qr_log("")
    _qr_log("=" * 60)
    _qr_log(f"Scanned at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _qr_log(f"Image size : {original.shape[1]}x{original.shape[0]} px")

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Build a list of image variants to try, from cheapest to most aggressive
    variants = [
        ("original colour",    original),
        ("grayscale",          gray),
        ("upscaled 2x",        cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
        ("upscaled 3x",        cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)),
        ("otsu threshold",     cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("adaptive threshold", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY, 11, 2)),
        ("upscale+otsu",       cv2.threshold(
                                   cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                                   0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
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

def extract_images_from_pdf(pdf_data):
    """Extracts images from each page of a PDF file."""
    images = []
    pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
    
    for page_num in range(len(pdf_document)):
        pix = pdf_document[page_num].get_pixmap()
        image_data = pix.tobytes("png")  # Convert to PNG byte format
        images.append(image_data)
    
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


#-----------------------------------------------------------------------INVOICE PROCESSING USING OPENAI-----------------------------------------------------------------------
def process_invoice(image_data, selected_fields=None, custom_fields=None):
    """Extract structured data from an invoice image using Gemini 2.5 Flash.

    selected_fields -> list of PREDEFINED_FIELDS dicts the user chose to extract.
                       Defaults to ALL predefined fields (= original behaviour).
    custom_fields   -> list of extra field names the user typed in the UI.
    """
    # Default to every predefined field so old behaviour is preserved when
    # the caller passes nothing.
    if selected_fields is None:
        selected_fields = PREDEFINED_FIELDS
    if custom_fields is None:
        custom_fields = []

    # try:
    image = Image.open(io.BytesIO(image_data))

    # Build the prompt dynamically from the chosen fields
    prompt = build_invoice_prompt(selected_fields, custom_fields)

    response = gemini_model.generate_content([prompt, image])

    # Extract and clean response
    response_text = response.text.strip()
    print(response_text)
    # Remove the backticks and "json" label
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        cleaned_json = match.group(1).strip()
    else:
        cleaned_json = response_text.strip()

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

    # ✅ Line Items — only if the user asked for them
    if "Line_Items" in selected_keys:
        line_items = find_value(invoice_data, ["Line Items", "Line_Items", "LineItems", "Line Item", "Line_Item"], [])
        new_invoice_data["Line_Items"] = []
        if isinstance(line_items, list):
            new_invoice_data["Line_Items"] = [
                {
                    "Item_Name": next(
                        (item[key] for key in ["Item Name", "Item_Name", "ItemName"] if key in item), "Unknown"
                    ),
                    "Item_Description": next(
                        (item[key] for key in ["Item Description", "Item_Description", "ItemDescription"] if key in item), ""
                    ),
                    "Quantity": safe_float(next(
                        (item[key] for key in ["Quantity", "Qty", "QTY"] if key in item), 0
                    )),
                    "Unit_Price": safe_float(next(
                        (item[key] for key in ["Unit Price", "Unit_Price", "UnitPrice", "Price Per Unit"] if key in item), 0.0
                    )),
                    "Total_Price": safe_float(next(
                        (item[key] for key in ["Total Price", "Total_Price", "TotalPrice", "Line Total"] if key in item), 0.0
                    ))
                }
                for item in line_items if isinstance(item, dict)
            ]

    # ✅ Custom user-defined fields — stored under their own label as the column name
    for cf in custom_fields:
        cf = cf.strip()
        if cf:
            new_invoice_data[cf] = find_value(invoice_data, [cf], "")

    qr_code_string = extract_qr_code(image_data)
    if qr_code_string:
        qr_data = decode_tlv_qr(qr_code_string)
    else:
        qr_data = None

    if qr_data:
        print(qr_data)
        new_invoice_data['QR_Code_Valid'] = True
        for key, qr_value in qr_data.items():
            if key in new_invoice_data and qr_value:
                # Supplier_Name: Arabic vs English causes false mismatches
                # Invoice_Date: QR stores a binary timestamp, Gemini reads the printed date correctly
                if key in ("Supplier_Name", "Invoice_Date"):
                    continue
                if new_invoice_data[key] != qr_value:
                    # Update other fields normally
                    print(f"Updating {key}: {new_invoice_data[key]} → {qr_value}")
                    new_invoice_data[key] = qr_value
    else:
        new_invoice_data['QR_Code_Valid'] = False
    print(new_invoice_data)
    return new_invoice_data  # Return consistent structured data

    # except json.JSONDecodeError:
    #     st.error("Error: Unable to decode JSON response.")
    # except Exception as e:
    #     st.error(f"Unexpected error: {e}")

    return None  # Return None if an error occurs


#--------------------------------------------------------------------------CHATBOT INTEGRATION--------------------------------------------------------------------------
from datetime import datetime
from dateutil import parser
# import groq
# from langchain_groq import ChatGroq
# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# groq_client = groq.Groq(api_key = GROQ_API_KEY)
# groq_llm = ChatGroq(
#     model = "llama-3.1-8b-instant",
#     temperature = 0.2,
# )


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
    if "last year" or "previous year" in query.lower():
        return now.year - 1
    elif "this year" or "current year" in query.lower():
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
st.sidebar.image('JFF-LOGO-White-removebg.png',width = 100)
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
        uploaded_files = st.file_uploader("Upload Invoices (PNG, JPG, PDF)", type=["png", "jpg", "pdf"], accept_multiple_files=True)
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

                for uploaded_file in uploaded_files:
                    file_data = uploaded_file.read()

                    if uploaded_file.type == "application/pdf":
                        split_invoices = (pdf_type == "Multiple Single-Page Invoices")
                        images = extract_images_from_pdf(file_data)

                        if not split_invoices:
                            images = [merge_images_vertically(images)]
                    else:
                        images = [file_data]

                    for image_data in images:
                        invoice_data = process_invoice(image_data, selected_fields, custom_fields)

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

                # Save to session only
                st.session_state.projects[selected_project].extend(new_invoices)

                if new_invoices:
                    st.success(f"Processed {len(new_invoices)} invoice(s) successfully!")

                if repeated_invoices:
                    st.warning(
                        f"Skipped {len(repeated_invoices)} duplicate invoice(s): "
                        f"{', '.join(repeated_invoices)}"
                    )

        # Display Invoices in a Table
        if selected_project:
            missing_data_records = []
            invoices = st.session_state.projects.get(selected_project, [])
            if invoices:
                # Convert to DataFrame
                df = pd.DataFrame(invoices)

                # Drop MongoDB `_id` field
                df = df.drop(columns=["_id", "Line_Items"], errors="ignore")

                # Clean column names
                df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")

                # Convert amounts to numeric
                for col in ["Total_Amount", "VAT_Amount", "Amount_Before_VAT"]:
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

                # Display total amounts
                # total_amount = df["Total_Amount"].sum()
                # total_vat = df["VAT_Amount"].sum()

                # st.markdown(f"### **Total Amount: {total_amount:,.2f}**")
                # st.markdown(f"### **Total VAT: {total_vat:,.2f}**")

            else:
                st.warning("No invoices found for this project.")

            # Display Missing Data Table
            # if missing_data_records:
            #     st.markdown("### **Invoices with Missing Data**")
            #     missing_df = pd.DataFrame(missing_data_records)
            #     st.dataframe(missing_df)


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
                if st.button("Clear Chat History", use_container_width = True):
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
        # Generate Donut Chart for Supplier VAT Status
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