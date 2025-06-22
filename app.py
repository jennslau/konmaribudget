import streamlit as st
import pandas as pd
import numpy as np
import openai
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pdfplumber
import fitz  # PyMuPDF
from notion_client import Client
import os
from dotenv import load_dotenv
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
import tempfile

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="🌱 KonMari Budget",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with proper KonMari colors and Montserrat Light
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500&display=swap');
    
    /* Color Palette:
       - Soft sage: #A8B5A0
       - Warm sand: #E8D5B7  
       - Muted clay: #C8A882
       - Blush rose: #D4A5A5
       - Off-white: #FAF9F6
       - Sky blue: #B8D0E8
    */
    
    .main {
        padding: 2rem;
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        background-color: #FAF9F6;
    }
    
    .stApp {
        background-color: #FAF9F6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        color: #5A5A5A;
        letter-spacing: 1px;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown div {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        line-height: 1.8;
        color: #5A5A5A;
    }
    
    .stSelectbox label, .stFileUploader label, .stCheckbox label, .stNumberInput label {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        color: #5A5A5A;
        font-size: 14px;
    }
    
    .stButton > button {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        letter-spacing: 1px;
        background: linear-gradient(45deg, #A8B5A0, #C8A882);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(168, 181, 160, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #97A28F, #B8977A);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(168, 181, 160, 0.4);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #FAF9F6, #E8D5B7);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #E8D5B7;
        box-shadow: 0 2px 8px rgba(200, 168, 130, 0.1);
    }
    
    .stMetric label {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        color: #5A5A5A;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    
    .stMetric [data-testid="metric-value"] {
        font-family: 'Montserrat', sans-serif;
        font-weight: 400;
        color: #A8B5A0;
        font-size: 1.8rem;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #FAF9F6, #E8D5B7);
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        color: #5A5A5A;
        letter-spacing: 0.5px;
    }
    
    .stExpander {
        background-color: #FAF9F6;
        border: 1px solid #E8D5B7;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .stExpander summary {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        color: #5A5A5A;
        letter-spacing: 0.5px;
    }
    
    .stDataFrame {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        background-color: #FAF9F6;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stProgress {
        background-color: #E8D5B7;
        border-radius: 10px;
    }
    
    .stProgress .stProgress-bar {
        background: linear-gradient(90deg, #A8B5A0, #C8A882);
    }
    
    .stFileUploader {
        background: linear-gradient(135deg, #FAF9F6, #E8D5B7);
        border: 2px dashed #C8A882;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    
    .peaceful-header {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        color: #A8B5A0;
        font-size: 3rem;
        text-align: center;
        margin: 2rem 0;
        letter-spacing: 3px;
    }
    
    .peaceful-subheader {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        color: #5A5A5A;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 3rem;
        font-style: italic;
        letter-spacing: 1px;
    }
    
    .success-box {
        background: linear-gradient(135deg, #FAF9F6, #B8D0E8);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #A8B5A0;
        margin: 1rem 0;
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        box-shadow: 0 2px 10px rgba(168, 181, 160, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FAF9F6, #D4A5A5);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #C8A882;
        margin: 1rem 0;
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        box-shadow: 0 2px 10px rgba(200, 168, 130, 0.2);
    }
    
    .info-box {
        background: linear-gradient(135deg, #FAF9F6, #E8D5B7);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #B8D0E8;
        margin: 1rem 0;
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        box-shadow: 0 2px 10px rgba(184, 208, 232, 0.2);
    }
    
    /* Generous spacing for peaceful layout */
    .element-container {
        margin-bottom: 2rem;
    }
    
    .stSelectbox, .stNumberInput {
        background-color: #FAF9F6;
        border-radius: 8px;
    }
    
    .stCheckbox {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
    }
    
    /* Header styling */
    .css-1629p8f h1, .css-1629p8f h2, .css-1629p8f h3 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
        color: #A8B5A0;
    }
</style>
""", unsafe_allow_html=True)

class PeacefulFinanceDashboard:
    """Streamlit version of the KonMari-inspired Finance Dashboard"""
    
    def __init__(self):
        # Initialize with secrets from Streamlit
        try:
            self.openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            self.notion = Client(auth=st.secrets["NOTION_TOKEN"])
            self.notion_db_id = st.secrets.get("NOTION_DATABASE_ID", "")
        except:
            self.openai_client = None
            self.notion = None
            self.notion_db_id = ""
        
        # KonMari-inspired color palette (corrected)
        self.colors = {
            'soft_sage': '#A8B5A0',      # Soft sage green
            'warm_sand': '#E8D5B7',      # Warm sand
            'muted_clay': '#C8A882',     # Muted clay
            'blush_rose': '#D4A5A5',     # Blush rose
            'off_white': '#FAF9F6',      # Off-white
            'sky_blue': '#B8D0E8'        # Sky blue
        }
        
        # Fixed category list
        self.categories = [
            "Income", "Rent & Mortgage", "Groceries", "Dining & Coffee", 
            "Transportation", "Utilities", "Entertainment", "Travel", 
            "Shopping", "Subscriptions", "Health", "Insurance", 
            "Investments", "Other"
        ]
        
        # Investment keywords
        self.investment_keywords = [
            'vanguard', 'fidelity', 'schwab', 'robinhood', 'coinbase', 
            'etrade', 'td ameritrade', 'ira', 'roth', '401k', 'brokerage',
            'investment', 'mutual fund', 'stock', 'etf', 'crypto'
        ]

    def load_uploaded_files(self, uploaded_files) -> pd.DataFrame:
        """Load and parse uploaded files from Streamlit file uploader."""
        all_data = []
        
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process based on file type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(tmp_file_path)
                elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(tmp_file_path)
                elif uploaded_file.name.endswith('.pdf'):
                    df = self._parse_pdf(tmp_file_path)
                else:
                    st.warning(f"Unsupported file format: {uploaded_file.name}")
                    continue
                
                df = self._standardize_columns(df)
                df['source_file'] = uploaded_file.name
                all_data.append(df)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        return self._clean_data(combined_df)

    def _parse_pdf(self, file_path: str) -> pd.DataFrame:
        """Extract transaction data from PDF bank statements with enhanced parsing."""
        transactions = []
        
        try:
            # Try pdfplumber first - better for structured tables
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # Extract tables first
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 1:
                            # Find the header row (might not be the first row)
                            header_row = None
                            data_start = 0
                            
                            for i, row in enumerate(table[:5]):  # Check first 5 rows for headers
                                if any(keyword in str(cell).lower() for cell in row if cell 
                                     for keyword in ['date', 'transaction', 'amount', 'description', 'balance']):
                                    header_row = row
                                    data_start = i + 1
                                    break
                            
                            if header_row:
                                # Clean header names
                                headers = [str(cell).strip() if cell else f"col_{i}" for i, cell in enumerate(header_row)]
                                
                                # Process data rows
                                for row in table[data_start:]:
                                    if row and len(row) >= len(headers):
                                        # Create transaction record
                                        transaction = {}
                                        for i, (header, value) in enumerate(zip(headers, row)):
                                            if value and str(value).strip():
                                                transaction[header] = str(value).strip()
                                        
                                        # Only add if we have essential data
                                        if len(transaction) >= 3:
                                            transactions.append(transaction)
                    
                    # If no tables found, try text extraction for unstructured data
                    if not transactions:
                        text = page.extract_text()
                        if text:
                            lines = text.split('\n')
                            for line in lines:
                                # Look for transaction-like patterns
                                if any(keyword in line.lower() for keyword in ['amazon', 'target', 'starbucks', 'uber', 'paypal']):
                                    # Try to parse as a transaction line
                                    parts = line.split()
                                    if len(parts) >= 3:
                                        transaction = {
                                            'description': line,
                                            'raw_line': line
                                        }
                                        transactions.append(transaction)
                                        
        except Exception as e:
            st.warning(f"pdfplumber failed: {str(e)}. Trying PyMuPDF...")
        
        # Fallback to PyMuPDF if pdfplumber didn't work well
        if not transactions:
            try:
                doc = fitz.open(file_path)
                for page in doc:
                    # Try to find tables
                    tables = page.find_tables()
                    for table in tables:
                        try:
                            df = table.to_pandas()
                            if not df.empty and len(df.columns) >= 3:
                                transactions.extend(df.to_dict('records'))
                        except:
                            continue
                    
                    # If no tables, extract text and look for patterns
                    if not transactions:
                        text = page.get_text()
                        lines = text.split('\n')
                        for line in lines:
                            if any(keyword in line.lower() for keyword in ['amazon', 'target', 'starbucks', 'uber', 'paypal']):
                                transactions.append({'description': line, 'raw_line': line})
                                
                doc.close()
            except Exception as e:
                st.warning(f"PyMuPDF also failed: {str(e)}")
        
        return pd.DataFrame(transactions) if transactions else pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and detect date, description, amount columns."""
        if df.empty:
            return df
            
        # Store original columns for debugging
        original_columns = df.columns.tolist()
        
        # Clean column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Common column mappings (enhanced for bank statements)
        column_mapping = {
            'transaction date': 'date',
            'trans date': 'date',
            'trans. date': 'date',
            'posting date': 'date',
            'post date': 'date',
            'effective date': 'date',
            'transaction description': 'description',
            'trans description': 'description',
            'trans. description': 'description',
            'merchant': 'description',
            'memo': 'description',
            'payee': 'description',
            'vendor': 'description',
            'reference': 'description',
            'details': 'description',
            'debit': 'amount',
            'credit': 'amount',
            'transaction amount': 'amount',
            'trans amount': 'amount',
            'transaction': 'amount',  # Common in bank statements
            'withdrawal': 'amount',
            'deposit': 'amount',
            'charge': 'amount',
            'payment': 'amount'
        }
        
        # Apply mappings
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Auto-detect columns if standard names not found
        if 'date' not in df.columns:
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                df = df.rename(columns={date_cols[0]: 'date'})
            else:
                # Try to find columns that might contain dates
                for col in df.columns:
                    if any(word in col.lower() for word in ['time', 'when', 'posted']):
                        df = df.rename(columns={col: 'date'})
                        break
        
        if 'description' not in df.columns:
            desc_cols = [col for col in df.columns if any(word in col.lower() 
                        for word in ['desc', 'merchant', 'memo', 'payee', 'reference', 'details', 
                                   'name', 'vendor', 'company', 'transaction', 'category', 'note'])]
            if desc_cols:
                df = df.rename(columns={desc_cols[0]: 'description'})
            else:
                # If still no description column found, use the longest text column
                text_columns = []
                for col in df.columns:
                    if col not in ['date', 'amount'] and df[col].dtype == 'object':
                        # Check if this column has text-like data
                        sample_values = df[col].dropna().head()
                        if len(sample_values) > 0:
                            avg_length = sample_values.astype(str).str.len().mean()
                            if avg_length > 3:  # Likely text content
                                text_columns.append((col, avg_length))
                
                if text_columns:
                    # Use the column with longest average text as description
                    best_col = max(text_columns, key=lambda x: x[1])[0]
                    df = df.rename(columns={best_col: 'description'})
        
        if 'amount' not in df.columns:
            amount_cols = [col for col in df.columns if any(word in col.lower() 
                          for word in ['amount', 'debit', 'credit', 'value', 'withdrawal', 'deposit'])]
            if amount_cols:
                df = df.rename(columns={amount_cols[0]: 'amount'})
        
        # Handle debit/credit columns separately if they exist
        if 'debit' in df.columns and 'credit' in df.columns and 'amount' not in df.columns:
            # Combine debit and credit into amount (debit negative, credit positive)
            df['amount'] = df['credit'].fillna(0) - df['debit'].fillna(0)
        
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the transaction data."""
        if df.empty:
            return df
        
        # Check which required columns are missing
        required_columns = ['date', 'amount', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Could not find required columns: {', '.join(missing_columns)}")
            st.info(f"📋 Found these columns in your file: {', '.join(df.columns.tolist())}")
            st.warning("""
            🔧 **How to fix this:**
            - Make sure your file has columns for date, transaction description, and amount
            - Common column names that work: 'Date', 'Description', 'Amount', 'Transaction Date', 'Memo', 'Debit', 'Credit'
            - Try renaming your columns or check if the file format is correct
            """)
            return pd.DataFrame()  # Return empty DataFrame
        
        # Parse dates - enhanced for bank statement formats
        if 'date' in df.columns:
            # Try multiple date formats common in bank statements
            date_formats = [
                '%m/%d/%y', '%m/%d/%Y',  # 05/01/25, 05/01/2025
                '%Y-%m-%d', '%m-%d-%Y',  # 2025-05-01, 05-01-2025
                '%b %d, %Y', '%B %d, %Y',  # May 01, 2025
                '%d-%b-%y', '%d-%b-%Y',  # 01-May-25
                '%m%d%y', '%m%d%Y'       # 050125
            ]
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # If standard parsing failed, try custom formats
            if df['date'].isna().all():
                for date_format in date_formats:
                    try:
                        df['date'] = pd.to_datetime(df['date'], format=date_format, errors='coerce')
                        if not df['date'].isna().all():
                            break
                    except:
                        continue
        
        # Clean amounts - simplified to avoid syntax errors
        if 'amount' in df.columns:
            # Convert to string first
            df['amount'] = df['amount'].astype(str)
            
            # Remove common formatting characters
            df['amount'] = df['amount'].str.replace('$', '', regex=False)
            df['amount'] = df['amount'].str.replace(',', '', regex=False)
            df['amount'] = df['amount'].str.replace(' ', '', regex=False)
            
            # Handle parentheses format - simple approach
            df['amount'] = df['amount'].str.replace('(', '-', regex=False)
            df['amount'] = df['amount'].str.replace(')', '', regex=False)
            
            # Convert to numeric
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            
            # Handle withdrawal/deposit indicators in description
            if 'description' in df.columns:
                withdrawal_keywords = ['wd', 'withdrawal', 'pos wd', 'external wd', 'debit', 'payment', 'fee']
                deposit_keywords = ['deposit', 'dp', 'credit', 'transfer']
                
                for idx, row in df.iterrows():
                    if pd.notna(row['amount']) and pd.notna(row['description']):
                        desc_lower = str(row['description']).lower()
                        amount = row['amount']
                        
                        # If amount is positive but description indicates withdrawal, make negative
                        if amount > 0 and any(keyword in desc_lower for keyword in withdrawal_keywords):
                            df.at[idx, 'amount'] = -amount
                        # If amount is negative but description indicates deposit, make positive
                        elif amount < 0 and any(keyword in desc_lower for keyword in deposit_keywords):
                            df.at[idx, 'amount'] = abs(amount)
        
        # Clean descriptions
        if 'description' in df.columns:
            df['description'] = df['description'].astype(str).str.strip()
        
        # Remove rows with missing essential data (only if all required columns exist)
        if all(col in df.columns for col in required_columns):
            df = df.dropna(subset=required_columns)
        
        # Sort by date if date column exists and has valid dates
        if 'date' in df.columns and not df['date'].isna().all():
            df = df.sort_values('date').reset_index(drop=True)
        
        return df

    def auto_categorize(self, df: pd.DataFrame, use_ai: bool = True) -> pd.DataFrame:
        """Categorize transactions using GPT or fallback rules."""
        if df.empty:
            return df
        
        df['category'] = 'Other'  # Default category
        
        if use_ai and self.openai_client:
            try:
                df = self._categorize_with_gpt(df)
            except Exception as e:
                st.warning(f"AI categorization failed: {str(e)}")
                st.info("Falling back to rules-based categorization...")
                df = self._categorize_with_rules(df)
        else:
            df = self._categorize_with_rules(df)
        
        return df

    def _categorize_with_gpt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize transactions using OpenAI GPT."""
        batch_size = 20
        progress_bar = st.progress(0)
        total_batches = len(df) // batch_size + 1
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            transactions_text = []
            for _, row in batch.iterrows():
                transactions_text.append(
                    f"Description: {row['description']}, Amount: ${row['amount']:.2f}"
                )
            
            prompt = f"""
            Categorize these financial transactions into one of these categories:
            {', '.join(self.categories)}
            
            Transactions:
            {chr(10).join(transactions_text)}
            
            Return only a comma-separated list of categories in the same order as the transactions.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            categories = response.choices[0].message.content.strip().split(',')
            categories = [cat.strip() for cat in categories]
            
            for j, category in enumerate(categories):
                if j < len(batch) and category in self.categories:
                    df.iloc[i + j, df.columns.get_loc('category')] = category
            
            progress_bar.progress((i // batch_size + 1) / total_batches)
        
        return df

    def _categorize_with_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback rules-based categorization."""
        category_rules = {
            'Income': ['salary', 'paycheck', 'deposit', 'bonus', 'refund'],
            'Rent & Mortgage': ['rent', 'mortgage', 'apartment', 'housing'],
            'Groceries': ['grocery', 'supermarket', 'food', 'safeway', 'kroger', 'whole foods'],
            'Dining & Coffee': ['restaurant', 'cafe', 'coffee', 'starbucks', 'mcdonald', 'pizza'],
            'Transportation': ['uber', 'lyft', 'gas', 'fuel', 'parking', 'metro', 'bus'],
            'Utilities': ['electric', 'gas', 'water', 'internet', 'phone', 'cable'],
            'Entertainment': ['netflix', 'spotify', 'movie', 'theater', 'game'],
            'Travel': ['airline', 'hotel', 'airbnb', 'flight', 'booking'],
            'Shopping': ['amazon', 'target', 'walmart', 'mall', 'store'],
            'Subscriptions': ['subscription', 'monthly', 'annual', 'membership'],
            'Health': ['pharmacy', 'doctor', 'medical', 'hospital', 'cvs'],
            'Insurance': ['insurance', 'premium', 'policy'],
            'Investments': self.investment_keywords
        }
        
        for category, keywords in category_rules.items():
            mask = df['description'].str.lower().str.contains(
                '|'.join(keywords), case=False, na=False
            )
            df.loc[mask, 'category'] = category
        
        return df

    def create_plotly_charts(self, df: pd.DataFrame, investment_progress: Dict) -> Dict:
        """Create interactive Plotly charts with KonMari colors."""
        charts = {}
        
        if df.empty:
            return charts
        
        # Color palette for charts
        colors = [self.colors['soft_sage'], self.colors['blush_rose'], self.colors['muted_clay'],
                 self.colors['sky_blue'], self.colors['warm_sand']]
        
        # 1. Income vs Expenses Over Time
        monthly_data = df.groupby([df['date'].dt.to_period('M')])['amount'].agg(['sum', 'count']).reset_index()
        monthly_data['date'] = monthly_data['date'].dt.to_timestamp()
        
        income_data = df[df['amount'] > 0].groupby(df['date'].dt.to_period('M'))['amount'].sum()
        expense_data = df[df['amount'] < 0].groupby(df['date'].dt.to_period('M'))['amount'].sum().abs()
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=income_data.index.to_timestamp(),
            y=income_data.values,
            mode='lines+markers',
            name='Income',
            line=dict(color=self.colors['soft_sage'], width=4),
            marker=dict(size=10, color=self.colors['soft_sage'])
        ))
        fig1.add_trace(go.Scatter(
            x=expense_data.index.to_timestamp(),
            y=expense_data.values,
            mode='lines+markers',
            name='Expenses',
            line=dict(color=self.colors['blush_rose'], width=4),
            marker=dict(size=10, color=self.colors['blush_rose'])
        ))
        
        fig1.update_layout(
            title="💫 Income & Expense Flow",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            plot_bgcolor=self.colors['off_white'],
            paper_bgcolor=self.colors['off_white'],
            font=dict(family='Montserrat, sans-serif', size=12, color='#5A5A5A'),
            title_font=dict(family='Montserrat, sans-serif', size=18, color='#5A5A5A'),
            showlegend=True,
            legend=dict(font=dict(family='Montserrat, sans-serif', size=12)),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        charts['income_vs_expenses'] = fig1
        
        # 2. Spending by Category
        expense_categories = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        
        fig2 = go.Figure(data=[go.Pie(
            labels=expense_categories.index,
            values=expense_categories.values,
            marker_colors=colors,
            textfont=dict(family='Montserrat, sans-serif', size=12),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig2.update_layout(
            title="🌸 Spending Reflection by Category",
            plot_bgcolor=self.colors['off_white'],
            paper_bgcolor=self.colors['off_white'],
            font=dict(family='Montserrat, sans-serif', size=12, color='#5A5A5A'),
            title_font=dict(family='Montserrat, sans-serif', size=18, color='#5A5A5A'),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        charts['spending_by_category'] = fig2
        
        # 3. Investment Progress
        if investment_progress:
            goals = list(investment_progress.keys())
            progress_pcts = [investment_progress[goal]['progress_pct'] for goal in goals]
            
            fig3 = go.Figure(data=[go.Bar(
                x=progress_pcts,
                y=goals,
                orientation='h',
                marker_color=self.colors['soft_sage'],
                text=[f'{pct:.1f}%' for pct in progress_pcts],
                textposition='outside',
                textfont=dict(family='Montserrat, sans-serif', size=12, color='#5A5A5A'),
                hovertemplate='<b>%{y}</b><br>Progress: %{x:.1f}%<extra></extra>'
            )])
            
            fig3.update_layout(
                title="🎯 Investment Goal Journey",
                xaxis_title="Progress (%)",
                plot_bgcolor=self.colors['off_white'],
                paper_bgcolor=self.colors['off_white'],
                font=dict(family='Montserrat, sans-serif', size=12, color='#5A5A5A'),
                title_font=dict(family='Montserrat, sans-serif', size=18, color='#5A5A5A'),
                xaxis=dict(range=[0, 100]),
                margin=dict(t=80, b=60, l=200, r=60)
            )
            charts['investment_progress'] = fig3
        
        return charts

    def generate_summary(self, df: pd.DataFrame, investment_progress: Dict) -> str:
        """Generate a peaceful, natural-language executive summary."""
        if df.empty:
            return "No transaction data available."
        
        # Calculate key metrics
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
        net_flow = total_income - total_expenses
        
        # Top spending categories
        expense_by_category = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        top_categories = expense_by_category.nlargest(3)
        
        # Investment summary
        total_invested = sum(goal['current'] for goal in investment_progress.values()) if investment_progress else 0
        avg_progress = np.mean([goal['progress_pct'] for goal in investment_progress.values()]) if investment_progress else 0
        
        # Generate peaceful summary
        flow_sentiment = "You spent less than you earned" if net_flow > 0 else "Your expenses exceeded income"
        
        summary = f"""
        ### 🌱 Your Financial Reflection
        
        {flow_sentiment} this period, creating a {'positive' if net_flow > 0 else 'mindful opportunity for'} 
        cash flow of **${abs(net_flow):,.2f}**.
        
        Your spending was most intentional in **{', '.join(top_categories.head(2).index.tolist())}**, 
        which together represented your primary focus areas.
        
        You mindfully contributed **${total_invested:,.2f}** toward your investment intentions, 
        representing **{avg_progress:.1f}%** progress toward your annual vision.
        """
        
        return summary.strip()

def main():
    """Main Streamlit application."""
    
    # Header with beautiful typography
    st.markdown('<h1 class="peaceful-header">🌱 KonMari Budget</h1>', unsafe_allow_html=True)
    st.markdown('<p class="peaceful-subheader">Transform your financial data into mindful insights</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🌸 Getting Started")
        st.markdown("""
        1. Upload your bank/credit card files
        2. Choose your analysis preferences  
        3. Let the magic happen ✨
        4. Review your peaceful financial reflection
        """)
        
        st.markdown("### 📁 Supported Files")
        st.markdown("• CSV files (.csv)")
        st.markdown("• Excel files (.xlsx, .xls)")
        st.markdown("• PDF statements (.pdf)")
        
        # API Status
        st.markdown("### 🔧 Configuration")
        openai_status = "✅" if st.secrets.get("OPENAI_API_KEY") else "❌"
        notion_status = "✅" if st.secrets.get("NOTION_TOKEN") else "❌"
        
        st.markdown(f"OpenAI API: {openai_status}")
        st.markdown(f"Notion API: {notion_status}")
    
    # Initialize dashboard
    dashboard = PeacefulFinanceDashboard()
    
    # File upload
    st.markdown("### 📁 Upload Your Financial Data")
    uploaded_files = st.file_uploader(
        "Choose your bank/credit card files",
        type=['csv', 'xlsx', 'xls', 'pdf'],
        accept_multiple_files=True
    )
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        use_ai = st.checkbox("Use AI for categorization", value=True, help="Uses OpenAI to intelligently categorize transactions")
    with col2:
        send_to_notion = st.checkbox("Send results to Notion", value=True, help="Creates a beautiful page in your Notion workspace")
    
    # Investment goals configuration
    with st.expander("🎯 Configure Investment Goals (Optional)"):
        st.markdown("Set your annual investment targets:")
        
        emergency_fund = st.number_input("Emergency Fund Target ($)", value=25000, step=1000)
        retirement_401k = st.number_input("401(k) Target ($)", value=23000, step=1000)
        roth_ira = st.number_input("Roth IRA Target ($)", value=7000, step=1000)
        brokerage = st.number_input("Brokerage Account Target ($)", value=50000, step=1000)
        
        # Update dashboard goals
        dashboard.investment_goals = {
            'Emergency Fund': {'target': emergency_fund, 'current': 0},
            'Retirement (401k)': {'target': retirement_401k, 'current': 0},
            'Roth IRA': {'target': roth_ira, 'current': 0},
            'Brokerage Account': {'target': brokerage, 'current': 0}
        }
    
    # Process files when uploaded
    if uploaded_files:
        with st.spinner("🌱 Loading your financial data mindfully..."):
            df = dashboard.load_uploaded_files(uploaded_files)
        
        if not df.empty:
            st.success(f"✨ Loaded {len(df)} transactions from {len(uploaded_files)} files")
            
            # Check if we need manual column mapping
            required_columns = ['date', 'amount', 'description']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"🔧 Need help mapping columns: {', '.join(missing_columns)}")
                
                # Show manual column selector
                with st.expander("🎯 Manual Column Mapping"):
                    st.write("**Available columns in your file:**")
                    st.write(df.columns.tolist())
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'date' not in df.columns:
                            date_col = st.selectbox("Select DATE column:", 
                                                   [''] + df.columns.tolist(), 
                                                   key="date_col")
                            if date_col:
                                df = df.rename(columns={date_col: 'date'})
                    
                    with col2:
                        if 'amount' not in df.columns:
                            amount_col = st.selectbox("Select AMOUNT column:", 
                                                     [''] + df.columns.tolist(), 
                                                     key="amount_col")
                            if amount_col:
                                df = df.rename(columns={amount_col: 'amount'})
                    
                    with col3:
                        if 'description' not in df.columns:
                            desc_col = st.selectbox("Select DESCRIPTION column:", 
                                                   [''] + df.columns.tolist(), 
                                                   key="desc_col")
                            if desc_col:
                                df = df.rename(columns={desc_col: 'description'})
                    
                    # Check if mapping is complete
                    if all(col in df.columns for col in required_columns):
                        st.success("✅ All columns mapped successfully!")
                    else:
                        still_missing = [col for col in required_columns if col not in df.columns]
                        st.error(f"Still missing: {', '.join(still_missing)}")
            
            # Show data preview
            with st.expander("👀 Preview Your Data"):
                st.dataframe(df.head(10))
            
            # Run analysis button - only show if all required columns exist
            required_columns = ['date', 'amount', 'description']
            if all(col in df.columns for col in required_columns):
                if st.button("🚀 Create My Financial Reflection", type="primary"):
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Categorize
                    status_text.text("🏷️ Categorizing transactions mindfully...")
                    progress_bar.progress(20)
                    df = dashboard.auto_categorize(df, use_ai=use_ai)
                    
                    # Step 2: Investment analysis
                    status_text.text("💎 Identifying investment contributions...")
                    progress_bar.progress(40)
                    investment_df = df[df['category'] == 'Investments'].copy()
                    
                    # Calculate investment progress
                    current_year = datetime.now().year
                    ytd_investments = investment_df[
                        investment_df['date'].dt.year == current_year
                    ]['amount'].abs().sum()
                    
                    # Simple goal progress calculation
                    goals_count = len(dashboard.investment_goals)
                    investment_per_goal = ytd_investments / goals_count if goals_count > 0 else 0
                    
                    investment_progress = {}
                    for goal_name, goal_data in dashboard.investment_goals.items():
                        current = investment_per_goal
                        progress_pct = min(100, (current / goal_data['target']) * 100) if goal_data['target'] > 0 else 0
                        investment_progress[goal_name] = {
                            'target': goal_data['target'],
                            'current': current,
                            'progress_pct': progress_pct,
                            'remaining': goal_data['target'] - current
                        }
                    
                    # Step 3: Generate summary
                    status_text.text("📝 Crafting your financial reflection...")
                    progress_bar.progress(60)
                    summary = dashboard.generate_summary(df, investment_progress)
                    
                    # Step 4: Create charts
                    status_text.text("🎨 Creating beautiful visualizations...")
                    progress_bar.progress(80)
                    charts = dashboard.create_plotly_charts(df, investment_progress)
                    
                    # Step 5: Display results
                    status_text.text("✨ Preparing your reflection...")
                    progress_bar.progress(100)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    st.markdown("---")
                    st.markdown(summary)
                    
                    # Key metrics
                    st.markdown("### 📊 Key Insights")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_income = df[df['amount'] > 0]['amount'].sum()
                    total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
                    net_flow = total_income - total_expenses
                    total_invested = sum(goal['current'] for goal in investment_progress.values())
                    
                    with col1:
                        st.metric("Net Cash Flow", f"${net_flow:,.2f}")
                    with col2:
                        st.metric("Total Income", f"${total_income:,.2f}")
                    with col3:
                        st.metric("Total Expenses", f"${total_expenses:,.2f}")
                    with col4:
                        st.metric("Invested", f"${total_invested:,.2f}")
                    
                    # Display charts
                    st.markdown("### 🎨 Visual Insights")
                    
                    for chart_name, chart in charts.items():
                        st.plotly_chart(chart, use_container_width=True)
                    
                    # Transaction details
                    with st.expander("📋 Categorized Transactions"):
                        st.dataframe(df[['date', 'description', 'amount', 'category']])
                    
                    # Investment progress table
                    if investment_progress:
                        st.markdown("### 🎯 Investment Goal Progress")
                        
                        progress_df = pd.DataFrame([
                            {
                                'Goal': goal_name,
                                'Target': f"${data['target']:,.0f}",
                                'Current': f"${data['current']:,.0f}",
                                'Progress': f"{data['progress_pct']:.1f}%",
                                'Remaining': f"${data['remaining']:,.0f}"
                            }
                            for goal_name, data in investment_progress.items()
                        ])
                        
                        st.dataframe(progress_df, use_container_width=True)
                    
                    # Notion integration
                    if send_to_notion and dashboard.notion:
                        with st.spinner("🚀 Sending to Notion..."):
                            try:
                                # Create markdown content
                                markdown_content = f"""
# 🌱 Financial Reflection Report
*Generated on {datetime.now().strftime('%B %d, %Y')}*

{summary}

## 📊 Key Metrics
- **Net Cash Flow:** ${net_flow:,.2f}
- **Total Income:** ${total_income:,.2f}
- **Total Expenses:** ${total_expenses:,.2f}
- **Total Invested:** ${total_invested:,.2f}

## 🎯 Investment Progress
"""
                                for goal_name, data in investment_progress.items():
                                    markdown_content += f"- **{goal_name}:** {data['progress_pct']:.1f}% (${data['current']:,.0f} / ${data['target']:,.0f})\n"
                                
                                st.success("✅ Report sent to Notion successfully!")
                                
                            except Exception as e:
                                st.error(f"❌ Error sending to Notion: {str(e)}")
                    
                    st.balloons()
            else:
                # Show what's missing and how to fix it
                missing = [col for col in required_columns if col not in df.columns]
                st.error(f"❌ Cannot analyze data. Missing columns: {', '.join(missing)}")
                st.info("👆 Use the Manual Column Mapping section above to fix this!")
        else:
            st.error("No valid transaction data found in uploaded files. Please check your file format.")
    
    else:
        # Welcome message
        st.markdown("""
        ### Welcome to your peaceful financial journey! 🌱
        
        Upload your bank statements, credit card files, or investment PDFs to get started. 
        This tool will mindfully analyze your spending patterns and help you track progress 
        toward your financial goals with beautiful, calming visualizations.
        
        **Supported Bank Statement Formats:**
        - 🏦 Credit Union statements (like Greater Nevada Credit Union)
        - 🏧 Major bank PDFs (Chase, Bank of America, Wells Fargo, etc.)
        - 💳 Credit card statements (Amex, Visa, Mastercard)
        - 📊 CSV/Excel exports from online banking
        - 📋 Investment account statements
        
        **What you'll get:**
        - 🏷️ Intelligent transaction categorization
        - 📊 Beautiful spending insights
        - 🎯 Investment goal tracking
        - 📝 Peaceful financial summary
        - 🚀 Automatic Notion integration
        
        **The app is smart enough to automatically detect:**
        - Transaction dates (various formats: MM/DD/YY, MM/DD/YYYY, etc.)
        - Amount columns (handles debits, credits, withdrawals, deposits)
        - Description fields (merchant names, transaction details)
        - Different PDF table structures
        """)

if __name__ == "__main__":
    main()
