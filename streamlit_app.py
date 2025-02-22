import streamlit as st
import openai
import pandas as pd
import base64
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import re
import json

# OpenAI API Key
OPENAI_API_KEY = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize session state for projects if not already initialized
if "projects" not in st.session_state:
    st.session_state.projects = {}  # Dictionary to store project data

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

# Function to process invoice using OpenAI API
def process_invoice(image_data):
    """Extract structured data from an invoice image using GPT-4o."""
    try:
        base64_image = base64.b64encode(image_data).decode("utf-8")

        qr_content = extract_qr_code(image_data)
        qr_code_present = qr_content is not None
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that extracts structured data from invoices."},
                {"role": "user", "content": [
                    {"type": "text", "text": (
                        "Extract the following details from this invoice and return them in JSON format:\n"
                        "- Invoice Number\n- Invoice Date\n- Supplier Name\n- Supplier VAT\n"
                        "- Customer Name\n- Customer VAT\n"
                        "- Amount Before VAT (Subtotal)\n"
                        "- VAT Amount\n"
                        "- Total Amount After VAT\n"
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=500
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

        # Convert response to dictionary
        invoice_data["QR Code Present"] = qr_code_present
        print(invoice_data)
        return invoice_data

    except json.JSONDecodeError:
        st.error("Error: Unable to decode JSON response.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

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

if selected_project:
    st.subheader(f"Project: {selected_project}")

    # Upload multiple invoices
    uploaded_files = st.file_uploader("Upload Invoices (PNG, JPG)", type=["png", "jpg"], accept_multiple_files=True)

    if st.button("Process Invoices"):
        if uploaded_files:
            existing_invoices = {inv["Invoice Number"] for inv in st.session_state.projects[selected_project]}  # Track existing invoice numbers
            new_invoices = []
            repeated_invoices = []

            for uploaded_file in uploaded_files:
                # Read file as binary
                file_data = uploaded_file.read()

                # Process invoice
                invoice_data = process_invoice(file_data)

                if invoice_data:
                    total_amount_key = "Total Amount After VAT" if "Total Amount After VAT" in invoice_data else "Total_Amount_After_VAT"
                    invoice_data["Total_Amount"] = invoice_data.pop(total_amount_key, None)
                    invoice_number = invoice_data.get("Invoice Number")

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

        # Convert amounts to numeric for summation
        for col in ["Total_Amount", "VAT_Amount", "Amount_Before_VAT_(Subtotal)"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(",", "").astype(float)

        st.dataframe(df)

        # Display Total Amount and Total VAT at the bottom
        total_amount = df["Total_Amount"].sum()
        total_vat = df["VAT_Amount"].sum()

        st.markdown(f"### **Total Amount: {total_amount:.2f}**")
        st.markdown(f"### **Total VAT: {total_vat:.2f}**")
