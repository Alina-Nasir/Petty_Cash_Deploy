import streamlit as st
import openai
import pandas as pd
import base64
import cv2
import numpy as np
# from pyzbar.pyzbar import decode
import re
import json
import fitz  # PyMuPDF for PDF processing
import plotly.express as px  # Import Plotly for visualization

# OpenAI API Key
OPENAI_API_KEY = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize session state for projects if not already initialized
if "projects" not in st.session_state:
    st.session_state.projects = {}  # Dictionary to store project data

# Function to check if a QR code is present in the image
# def extract_qr_code(image_data):
#     """Detects and extracts QR code content from the image."""
#     nparr = np.frombuffer(image_data, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     qr_codes = decode(image)

#     # If QR codes are found, extract the first one
#     if qr_codes:
#         return qr_codes[0].data.decode("utf-8")  # Decoding QR content to string
    
#     return None  # No QR code found

def extract_images_from_pdf(pdf_data):
    """Extracts images from each page of a PDF file."""
    images = []
    pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
    
    for page_num in range(len(pdf_document)):
        pix = pdf_document[page_num].get_pixmap()
        image_data = pix.tobytes("png")  # Convert to PNG byte format
        images.append(image_data)
    
    return images

# Function to process invoice using OpenAI API
def process_invoice(image_data):
    """Extract structured data from an invoice image using GPT-4o and ensure consistent key formatting."""
    # try:
    base64_image = base64.b64encode(image_data).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system","content": (
                "You are an AI specialized in extracting structured data from invoices."
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

    print(new_invoice_data)
    return new_invoice_data  # Return consistent structured data

    # except json.JSONDecodeError:
    #     st.error("Error: Unable to decode JSON response.")
    # except Exception as e:
    #     st.error(f"Unexpected error: {e}")

    return None  # Return None if an error occurs


# Streamlit App Layout
st.title("Invoice Processing App")

# Sidebar: Add New Project
st.sidebar.header("Project Management")
project_name = st.sidebar.text_input("Enter New Project Name")

if st.sidebar.button("Add Project"):
    if project_name:
        if project_name not in st.session_state.projects:
            st.session_state.projects[project_name] = []  # Initialize project with empty invoice list
            st.sidebar.success(f"Project '{project_name}' added!")
        else:
            st.sidebar.warning("Project already exists!")
    else:
        st.sidebar.error("Project name cannot be empty.")

# Select Project Dropdown
selected_project = st.sidebar.selectbox("Select a Project", list(st.session_state.projects.keys()))

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
    if page_selection == "Project Overview":

        # Upload multiple invoices
        uploaded_files = st.file_uploader("Upload Invoices (PNG, JPG, PDF)", type=["png", "jpg", "pdf"], accept_multiple_files=True)

        if st.button("Process Invoices"):
            if uploaded_files:
                existing_invoices = {inv["Invoice_Number"] for inv in st.session_state.projects[selected_project]}  # Track existing invoice numbers
                new_invoices = []
                repeated_invoices = []

                for uploaded_file in uploaded_files:
                    # Read file as binary
                    file_data = uploaded_file.read()
                    if uploaded_file.type == "application/pdf":
                        images = extract_images_from_pdf(file_data)
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
                                    new_invoices.append(invoice_data)
                                    existing_invoices.add(invoice_number)  # Update existing invoices set

                # Save new invoices
                st.session_state.projects[selected_project].extend(new_invoices)

                # Success message
                if new_invoices:
                    st.success(f"Processed {len(new_invoices)} new invoice(s) successfully!")

                # Warning for duplicates
                if repeated_invoices:
                    st.warning(f"Skipped {len(repeated_invoices)} repeated invoice(s): {', '.join(repeated_invoices)}")

        # Display Invoices in a Table
        if st.session_state.projects[selected_project]:
            df = pd.DataFrame(st.session_state.projects[selected_project])
            df.columns = df.columns.str.replace(" ", "_")  # Replace spaces with underscores
            df.columns = df.columns.str.replace("-", "_")  # Replace hyphens with underscores

            # Drop duplicate column headers if they exist
            df = df.loc[:, ~df.columns.duplicated()]
            # ✅ Remove the "Line_Items" column if it exists
            df = df.drop(columns=["Line_Items"], errors="ignore")

            # Convert amounts to numeric for summation
            for col in ["Total_Amount", "VAT_Amount", "Amount_Before_VAT"]:
                if col in df.columns:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(",", "", regex=True)  # Remove commas
                        .str.replace(r"[^\d.]", "", regex=True)  # Remove non-numeric characters
                        .apply(lambda x: float(x) if re.match(r"^\d+(\.\d+)?$", x) else 0)
                    )

            st.dataframe(df)
            # Toggle Section for Viewing Line Items
            for invoice in st.session_state.projects[selected_project]:
                invoice_number = invoice.get("Invoice_Number", "Unknown")

                # Ensure Line Items is always a list
                if "Line_Items" in invoice:
                    if isinstance(invoice["Line_Items"], str):  # If stored as a string, convert it back to a list
                        try:
                            invoice["Line_Items"] = json.loads(invoice["Line_Items"])
                        except json.JSONDecodeError:
                            invoice["Line_Items"] = []  # If conversion fails, set an empty list

                if "Line_Items" in invoice and invoice["Line_Items"]:
                    with st.expander(f"📌 View Line Items for Invoice: {invoice_number}"):
                        line_items_df = pd.DataFrame(invoice["Line_Items"])
                        line_items_df.columns = line_items_df.columns.str.replace(" ", "_").str.replace("-", "_")  # Clean column names

                        # Convert numerical columns to correct data type
                        numeric_columns = ["Quantity", "Unit_Price", "Total_Price"]
                        for col in numeric_columns:
                            if col in line_items_df.columns:
                                line_items_df[col] = pd.to_numeric(line_items_df[col], errors="coerce")

                        st.dataframe(line_items_df, use_container_width=True)




            # Display Total Amount and Total VAT at the bottom
            total_amount = df["Total_Amount"].sum()
            total_vat = df["VAT_Amount"].sum()

            st.markdown(f"### **Total Amount: {total_amount:.2f}**")
            st.markdown(f"### **Total VAT: {total_vat:.2f}**")

            # Display Missing Data Table
            if st.session_state[f"{selected_project}_missing_data_records"]:
                st.markdown("### **Invoices with Missing Data**")
                missing_df = pd.DataFrame(st.session_state[f"{selected_project}_missing_data_records"])
                st.dataframe(missing_df)

    elif page_selection == "Analytics":
        st.title("📊 Project Analytics")
        # Generate Donut Chart for Supplier VAT Status
        total_invoices = len(st.session_state.projects[selected_project])  # Total invoices in the project
        supplier_vat_missing_count = st.session_state[f"{selected_project}_supplier_vat_missing_count"]
        invoices_with_supplier_vat = total_invoices - supplier_vat_missing_count  # Those with Supplier VAT

        if total_invoices > 0:
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
            bool(inv.get("QR_Code_Present") in [True, "True", 1]) 
            for inv in st.session_state.projects[selected_project]
        )

        qr_code_missing_count = total_invoices - qr_code_present_count  # Invoices without a QR code

        if total_invoices > 0:
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