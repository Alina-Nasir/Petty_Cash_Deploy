import streamlit as st
import openai
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


#--------------------------------------------------------------------------API KEY INITIALIZATIONS--------------------------------------------------------------------------
# OpenAI API Key
OPENAI_API_KEY = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)
MONGO_URI = st.secrets["MONGO_URI"]  # Store this in Streamlit Secrets
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["PettyCash"]  # Connect to the database
invoice_collection = db["Invoice"]
projects_coll = db["Project"]

# Initialize session state for projects if not already initialized
if "projects" not in st.session_state:
    st.session_state.projects = {}  # Dictionary to store project data


#--------------------------------------------------------------------------IMAGE DATA EXTRACTION--------------------------------------------------------------------------
# Function to check if a QR code is present in the image
def extract_qr_code(image_data):
    """Detects and extracts QR code content from the image."""
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    qr_codes = decode(image)

    # If QR codes are found, extract the first one
    if qr_codes:
        return qr_codes[0].data.decode("utf-8")  # Decoding QR content to string
    
    return None  # No QR code found

def decode_tlv_qr(qr_string):
    """
    Decodes the extracted QR code data (Base64-encoded TLV format) 
    used in E-Invoice QR Reader KSA.
    """
    try:
        qr_bytes = base64.b64decode(qr_string)
        fields = []
        i = 0
        def to_float(value):
            try:
                return float(value) if value else None
            except ValueError:
                return None  # If conversion fails, return None instead of crashing
        
        while i < len(qr_bytes):
            tag = qr_bytes[i]
            length = qr_bytes[i + 1]
            value = qr_bytes[i + 2 : i + 2 + length].decode('utf-8')
            fields.append(value)
            i += 2 + length

        return {
            "Supplier_Name": fields[0] if len(fields) > 0 else None,
            "Supplier_VAT": fields[1] if len(fields) > 1 else None,
            "Invoice_Date": fields[2] if len(fields) > 2 else None,
            "Total_Amount_After_VAT": to_float(fields[3]) if len(fields) > 3 else None,
            "VAT_Amount": to_float(fields[4]) if len(fields) > 4 else None
        }
    except Exception as e:
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
def process_invoice(image_data):
    """Extract structured data from an invoice image using GPT-4o and ensure consistent key formatting."""
    # try:
    base64_image = base64.b64encode(image_data).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system","content": (
                "You are an AI specialized in extracting structured data from invoices."
                "The invoice may contain text in English or Arabic, or both."
                "The supplier name is company that issued invoice."
                "Your response must always be a valid JSON object, formatted exactly as follows:"
                "\n```json\n"
                "{"
                "\n  \"Invoice Number\": \"<string>\","
                "\n  \"Invoice Date\": \"<string>\","
                "\n  \"Supplier Name\": \"<string>\","
                "\n  \"Supplier VAT\": \"<string>\","
                "\n  \"Customer Name\": \"<string>\","
                "\n  \"Customer VAT\": \"<string>\","
                "\n  \"Amount Before VAT\": <float>,"
                "\n  \"VAT Amount\": <float>,"
                "\n  \"Total Amount After VAT\": <float>,"
                "\n  \"QR Code Present\": <boolean>,"
                "\n  \"Line Items\": ["
                "\n    {"
                "\n      \"Item Name\": \"<string>\","
                "\n      \"Item Description\": \"<string>\","
                "\n      \"Quantity\": <int>,"
                "\n      \"Unit Price\": <float>,"
                "\n      \"Total Price\": <float>"
                "\n    }"
                "\n  ]"
                "\n}"
                "\n```"
                "\nEnsure the JSON structure remains consistent and does not wrap data in extra keys like 'Invoice'.")},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    "Extract the following details from this invoice and return them in JSON format:\n"
                    "- Invoice Number\n- Invoice Date\n- Supplier Name\n- Supplier VAT\n"
                    "- Customer Name\n- Customer VAT\n"
                    "- Amount Before VAT\n- VAT Amount\n- Total Amount After VAT\n"
                    "- QR Code Present\n"
                    "Additionally, extract line items listed in the invoice. Each line item should include:\n"
                    "- Item Name\n- Item Description (if available)\n- Quantity\n- Unit Price\n- Total Price\n"
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=800
    )

    # Extract and clean response
    response_text = response.choices[0].message.content.strip()
    print(response_text)
    # Remove the backticks and "json" label
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        cleaned_json = match.group(1).strip()
    else:
        cleaned_json = response_text.strip()

    # Convert JSON string to dictionary
    invoice_data = json.loads(cleaned_json)

    def get_value(keys, default="Unknown"):
        """Helper function to get value from multiple possible keys in invoice_data."""
        for key in keys:
            if key in invoice_data:
                return invoice_data[key]
        return default
    def safe_float(value, default=0.0):
        """Convert value to float, handling None or invalid values."""
        try:
            return float(value) if value is not None else default
        except ValueError:
            return default

    # ✅ Ensure consistent keys with default values
    new_invoice_data = {
        "Invoice_Number": get_value(["Invoice Number", "Invoice_Number", "InvoiceNumber"]),
        "Invoice_Date": get_value(["Invoice Date", "Invoice_Date", "InvoiceDate"]),
        "Supplier_Name": get_value(["Supplier Name", "Supplier_Name","SupplierName"]),
        "Supplier_VAT": get_value(["Supplier VAT", "Supplier_VAT", "SupplierVAT"]),
        "Customer_Name": get_value(["Customer Name", "Customer_Name","CustomerName"]),
        "Customer_VAT": get_value(["Customer VAT", "Customer_VAT","CustomerVAT"]),
        "Amount_Before_VAT": get_value(["Amount Before VAT", "Amount_Before_VAT, AmountBeforeVAT"], "0.00"),
        "VAT_Amount": get_value(["VAT Amount", "VAT_Amount", "VATAmount"], "0.00"),
        "Total_Amount_After_VAT": get_value(["Total Amount After VAT", "Total_Amount_After_VAT","TotalAmountAfterVAT"], "0.00"),
        "QR_Code_Present": get_value(["QR Code Present", "QR_Code_Present", "QRCodePresent"], False),
        "Line_Items": []
    }

    # ✅ Handle multiple variations of "Line Items"
    line_items_keys = ["Line Items", "Line_Items","LineItems","LineItem" "Line Item", "Line_Item"]
    line_items = next((invoice_data[key] for key in line_items_keys if key in invoice_data), [])

    # ✅ Process line items if they exist
    if isinstance(line_items, list):
        new_invoice_data["Line_Items"] = [
            {
                "Item_Name": next(
                    (item[key] for key in ["Item Name", "Item_Name", "ItemName" ] if key in item), "Unknown"
                ),
                "Item_Description": next(
                    (item[key] for key in ["Item Description", "Item_Description", "ItemDescription"] if key in item), ""
                ),
                "Quantity": safe_float(next(
                    (item[key] for key in ["Quantity", "Qty", "QTY"] if key in item), 0
                )),
                "Unit_Price": safe_float(next(
                    (item[key] for key in ["Unit Price", "Unit_Price","UnitPrice", "Price Per Unit"] if key in item), 0.0
                )),
                "Total_Price": safe_float(next(
                    (item[key] for key in ["Total Price", "Total_Price", "TotalPrice","Line Total"] if key in item), 0.0
                ))
            }
            for item in line_items
        ]
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
                if key == "Supplier_Name":
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
import groq
from langchain_groq import ChatGroq
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_client = groq.Groq(api_key = GROQ_API_KEY)
groq_llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.2,
)


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


def llm_output(invoices, query, llm=groq_llm):
    data = ""
    for invoice in invoices[0:len(invoices)]:
        line_items = invoice.get('Line_Items', [])
        if not isinstance(line_items, list):
            # Handle if someone accidentally stored a bool or dict in the DB
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

    messages = [
        {
            'role': 'system', 'content': "You are an invoice data assistant that returns answers to the users queries in a structured manner, you are thorough and accurate"
        },
        {
            'role': 'user', 'content': f"""
                User question: {query}
                Here is the invoice data: {data}
                Now provide a concise summary...
            """
        }
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
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
#st.sidebar.image('JFF-LOGO-White-removebg.png',width = 100)
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

        # Upload multiple invoices
        uploaded_files = st.file_uploader("Upload Invoices (PNG, JPG, PDF)", type=["png", "jpg", "pdf"], accept_multiple_files=True)
        pdf_type = st.radio("Is your PDF file:", ["One Invoice (Multiple Pages)", "Multiple Single-Page Invoices"])
        if st.button("Process Invoices"):
            if uploaded_files:
                existing_invoices = {inv["Invoice_Number"] for inv in invoice_collection.find({"Project": selected_project})}  # Track existing invoice numbers
                new_invoices = []
                repeated_invoices = []

                for uploaded_file in uploaded_files:
                    # Read file as binary
                    file_data = uploaded_file.read()
                    if uploaded_file.type == "application/pdf":
                        split_invoices = (pdf_type == "Multiple Single-Page Invoices")
                        images = extract_images_from_pdf(file_data)
                        if not split_invoices:
                            # If it's a single invoice spanning multiple pages, merge images
                            merged_invoice_image = merge_images_vertically(images)
                            images = [merged_invoice_image]
                    else:
                    # Process invoice
                        images = [file_data]
                    for image_data in images:
                            invoice_data = process_invoice(image_data)
                            if invoice_data:
                                total_amount_key = "Total Amount After VAT" if "Total Amount After VAT" in invoice_data else "Total_Amount_After_VAT"
                                invoice_data["Total_Amount"] = invoice_data.pop(total_amount_key, None)
                                invoice_number = invoice_data.get("Invoice_Number")

                                # Check for missing fields
                                missing_fields = [key for key, value in invoice_data.items() if value in (None, "", "N/A")]

                                if missing_fields:
                                    st.session_state[f"{selected_project}_missing_data_records"].append({
                                        "Invoice_Number": invoice_number,
                                        "Missing Fields": ", ".join(missing_fields)
                                    })

                                # Check if Supplier VAT is missing
                                supplier_vat_key = "Supplier VAT" if "Supplier VAT" in invoice_data else "Supplier_VAT"
                                if invoice_data.get(supplier_vat_key) in (None, "", "N/A"):
                                    st.session_state[f"{selected_project}_supplier_vat_missing_count"] += 1

                                if invoice_number in existing_invoices:
                                    repeated_invoices.append(invoice_number)
                                else:
                                    invoice_data["File Name"] = uploaded_file.name  # Track file name
                                    invoice_data["Project"] = selected_project
                                    new_invoices.append(invoice_data)
                                    existing_invoices.add(invoice_number)  # Update existing invoices set

                # Save new invoices
                st.session_state.projects[selected_project].extend(new_invoices)

                # Success message
                if new_invoices:
                    invoice_collection.insert_many(new_invoices)
                    st.success(f"Processed {len(new_invoices)} new invoice(s) successfully!")

                # Warning for duplicates
                if repeated_invoices:
                    st.warning(f"Skipped {len(repeated_invoices)} repeated invoice(s): {', '.join(repeated_invoices)}")

        # Display Invoices in a Table
        if selected_project:
            missing_data_records = []
            invoices = list(invoice_collection.find({"Project": selected_project}))
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
                total_amount = df["Total_Amount"].sum()
                total_vat = df["VAT_Amount"].sum()

                st.markdown(f"### **Total Amount: {total_amount:,.2f}**")
                st.markdown(f"### **Total VAT: {total_vat:,.2f}**")

            else:
                st.warning("No invoices found for this project.")

            # Display Missing Data Table
            if missing_data_records:
                st.markdown("### **Invoices with Missing Data**")
                missing_df = pd.DataFrame(missing_data_records)
                st.dataframe(missing_df)


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
        invoices = list(invoice_collection.find({"Project": selected_project}))
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