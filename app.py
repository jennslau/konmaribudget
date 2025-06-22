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
    
    .stMetric {
        background: linear-gradient(135deg, #FAF9F6, #E8D5B7);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #E8D5B7;
        box-shadow: 0 2px 8px rgba(200, 168, 130, 0.1);
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
        
        # KonMari-inspired color palette
        self.colors = {
            'soft_sage': '#A8B5A0',
            'warm_sand': '#E8D5B7',
            'muted_clay': '#C8A882',
            'blush_rose': '#D4A5A5',
            'off_white': '#FAF9F6',
            'sky_blue': '#B8D0E8'
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
            st.info(f"🔄 Processing: {uploaded_file.name}")
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process based on file type
                if uploaded_file.name.endswith('.csv'):
                    st.info("📊 Reading as CSV file...")
                    df = pd.read_csv(tmp_file_path)
                elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                    st.info("📊 Reading as Excel file...")
                    df = pd.read_excel(tmp_file_path)
                elif uploaded_file.name.endswith('.pdf'):
                    st.info("📄 Reading as PDF file...")
                    df = self._parse_pdf(tmp_file_path)
                else:
                    st.warning(f"Unsupported file format: {uploaded_file.name}")
                    continue
                
                if df.empty:
                    st.error(f"❌ No data extracted from {uploaded_file.name}")
                    continue
                
                st.success(f"✅ Raw data extracted: {len(df)} rows, {len(df.columns)} columns")
                st.info(f"📋 Raw columns: {list(df.columns)}")
                
                # Show sample of raw data
                if len(df) > 0:
                    st.info("🔍 Sample of raw data:")
                    st.dataframe(df.head(3))
                
                df = self._standardize_columns(df)
                df['source_file'] = uploaded_file.name
                all_data.append(df)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        if not all_data:
            st.error("❌ No valid data found in any uploaded files")
            return pd.DataFrame()
        
        st.success(f"🎉 Successfully processed {len(all_data)} files")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        st.info(f"📊 Combined data: {len(combined_df)} total transactions")
        return self._clean_data(combined_df)

    def _parse_pdf(self, file_path: str) -> pd.DataFrame:
        """Extract transaction data from PDF bank statements."""
        transactions = []
        
        st.info("🔍 Analyzing your PDF structure...")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    st.info(f"📄 Processing page {page_num + 1}...")
                    
                    # Extract tables first
                    tables = page.extract_tables()
                    
                    if tables:
                        st.info(f"Found {len(tables)} tables on page {page_num + 1}")
                        
                        for table_num, table in enumerate(tables):
                            if table and len(table) > 1:
                                st.info(f"Processing table {table_num + 1} with {len(table)} rows")
                                
                                # Look for headers
                                header_row = None
                                data_start = 0
                                
                                for i, row in enumerate(table[:10]):
                                    if row and any(cell for cell in row):
                                        row_text = ' '.join([str(cell).lower() if cell else '' for cell in row])
                                        if any(keyword in row_text for keyword in [
                                            'trans date', 'transaction', 'amount', 'description', 
                                            'balance', 'date', 'memo'
                                        ]):
                                            header_row = row
                                            data_start = i + 1
                                            st.success(f"✅ Found headers in row {i + 1}")
                                            break
                                
                                if header_row:
                                    headers = []
                                    for j, cell in enumerate(header_row):
                                        if cell and str(cell).strip():
                                            clean_header = str(cell).strip().lower()
                                            headers.append(clean_header)
                                        else:
                                            headers.append(f"col_{j}")
                                    
                                    st.info(f"📋 Detected columns: {headers}")
                                    
                                    # Process data rows
                                    valid_transactions = 0
                                    for row_idx, row in enumerate(table[data_start:], start=data_start):
                                        if row and len(row) >= len(headers):
                                            transaction = {}
                                            has_meaningful_data = False
                                            
                                            for header, value in zip(headers, row):
                                                if value and str(value).strip():
                                                    clean_value = str(value).strip()
                                                    transaction[header] = clean_value
                                                    if any(keyword in clean_value.lower() for keyword in [
                                                        'amazon', 'target', 'paypal', 'external', 'pos', 'deposit',
                                                        'may', 'apr', 'jun'
                                                    ]) or any(char.isdigit() for char in clean_value):
                                                        has_meaningful_data = True
                                            
                                            if len(transaction) >= 3 and has_meaningful_data:
                                                transaction['page'] = page_num + 1
                                                transactions.append(transaction)
                                                valid_transactions += 1
                                    
                                    st.info(f"✅ Extracted {valid_transactions} transactions from table {table_num + 1}")
                    else:
                        st.info("No structured tables found, trying text extraction...")
                        text = page.extract_text()
                        
                        if text:
                            lines = text.split('\n')
                            for line_num, line in enumerate(lines):
                                line_lower = line.lower()
                                if any(keyword in line_lower for keyword in [
                                    'amazon', 'target', 'paypal', 'external wd', 'pos wd'
                                ]) and any(char.isdigit() for char in line):
                                    transaction = {
                                        'raw_line': line,
                                        'description': line,
                                        'page': page_num + 1
                                    }
                                    transactions.append(transaction)
                        
        except Exception as e:
            st.error(f"PDF parsing failed: {str(e)}")
        
        st.info(f"📊 Total transactions found: {len(transactions)}")
        
        if transactions:
            st.info("🔍 Sample of extracted data:")
            sample_size = min(3, len(transactions))
            for i, trans in enumerate(transactions[:sample_size]):
                st.write(f"Transaction {i+1}: {trans}")
        
        return pd.DataFrame(transactions) if transactions else pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and detect date, description, amount columns."""
        if df.empty:
            return df
        
        # Clean column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Common column mappings
        column_mapping = {
            'transaction date': 'date',
            'trans date': 'date',
            'trans. date': 'date',
            'posting date': 'date',
            'description': 'description',
            'transaction': 'amount',
            'amount': 'amount',
            'debit': 'amount',
            'credit': 'amount'
        }
        
        st.info(f"🔍 Original columns found: {list(df.columns)}")
        
        # Apply mappings
        columns_mapped = []
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                columns_mapped.append(f"{old_name} → {new_name}")
        
        if columns_mapped:
            st.success(f"✅ Mapped columns: {', '.join(columns_mapped)}")
        
        # Auto-detect missing columns
        if 'date' not in df.columns:
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                df = df.rename(columns={date_cols[0]: 'date'})
                st.success(f"✅ Mapped '{date_cols[0]}' to 'date'")
        
        if 'description' not in df.columns:
            desc_cols = [col for col in df.columns if 'desc' in col.lower() or 'memo' in col.lower()]
            if desc_cols:
                df = df.rename(columns={desc_cols[0]: 'description'})
                st.success(f"✅ Mapped '{desc_cols[0]}' to 'description'")
        
        if 'amount' not in df.columns:
            amount_cols = [col for col in df.columns if any(word in col.lower() 
                          for word in ['amount', 'transaction', 'balance'])]
            if amount_cols:
                df = df.rename(columns={amount_cols[0]: 'amount'})
                st.success(f"✅ Mapped '{amount_cols[0]}' to 'amount'")
        
        st.info(f"🎯 Final columns: {list(df.columns)}")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the transaction data."""
        if df.empty:
            st.warning("📭 DataFrame is empty, nothing to clean")
            return df
        
        st.info(f"🧹 Cleaning data with columns: {list(df.columns)}")
        
        # Check which required columns are missing
        required_columns = ['date', 'amount', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info(f"📋 Available columns: {', '.join(df.columns.tolist())}")
            
            # Try to auto-detect missing columns
            st.info("🔍 Attempting to auto-detect missing columns...")
            
            # Auto-detect date column
            if 'date' not in df.columns:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        sample_values = df[col].dropna().head(10)
                        if len(sample_values) > 0:
                            date_patterns = ['may', 'apr', 'jun', '01', '02', '/', '-', '2024', '2025']
                            if any(pattern in str(val).lower() for val in sample_values for pattern in date_patterns):
                                df = df.rename(columns={col: 'date'})
                                st.success(f"✅ Auto-detected '{col}' as date column")
                                break
            
            # Auto-detect amount column
            if 'amount' not in df.columns:
                for col in df.columns:
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        numeric_count = 0
                        for val in sample_values:
                            val_str = str(val)
                            if any(char.isdigit() for char in val_str):
                                numeric_count += 1
                        
                        if numeric_count >= len(sample_values) * 0.7:
                            df = df.rename(columns={col: 'amount'})
                            st.success(f"✅ Auto-detected '{col}' as amount column")
                            break
            
            # Auto-detect description column
            if 'description' not in df.columns:
                for col in df.columns:
                    if col not in ['date', 'amount'] and df[col].dtype == 'object':
                        sample_values = df[col].dropna().head(10)
                        if len(sample_values) > 0:
                            desc_keywords = ['amazon', 'target', 'paypal', 'external', 'pos']
                            if any(keyword in str(val).lower() for val in sample_values for keyword in desc_keywords):
                                df = df.rename(columns={col: 'description'})
                                st.success(f"✅ Auto-detected '{col}' as description column")
                                break
            
            # If still missing, use first available text column for description
            if 'description' not in df.columns:
                text_cols = [col for col in df.columns if col not in ['date', 'amount'] and df[col].dtype == 'object']
                if text_cols:
                    df = df.rename(columns={text_cols[0]: 'description'})
                    st.success(f"✅ Using '{text_cols[0]}' as description column")
            
            # Re-check missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.warning(f"Still missing: {', '.join(missing_columns)}. Please use manual column mapping.")
                return df
        
        # Clean dates
        if 'date' in df.columns:
            st.info("📅 Cleaning date column...")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean amounts
        if 'amount' in df.columns:
            st.info("💰 Cleaning amount column...")
            try:
                # Convert to string
                df['amount'] = df['amount'].astype(str)
                
                # Remove dollar signs and commas
                df['amount'] = df['amount'].str.replace('$', '')
                df['amount'] = df['amount'].str.replace(',', '')
                df['amount'] = df['amount'].str.replace(' ', '')
                
                # Handle parentheses (negative numbers)
                df['amount'] = df['amount'].str.replace('(', '-')
                df['amount'] = df['amount'].str.replace(')', '')
                
                # Convert to numeric
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                
                st.success(f"✅ Cleaned {df['amount'].count()} valid amounts")
            except Exception as e:
                st.error(f"❌ Error cleaning amounts: {str(e)}")
        
        # Clean descriptions
        if 'description' in df.columns:
            st.info("📝 Cleaning description column...")
            df['description'] = df['description'].astype(str).str.strip()
            st.success(f"✅ Cleaned {df['description'].count()} descriptions")
        
        # Remove rows with missing data
        if all(col in df.columns for col in required_columns):
            initial_count = len(df)
            df = df.dropna(subset=required_columns)
            removed_count = initial_count - len(df)
            if removed_count > 0:
                st.info(f"🗑️ Removed {removed_count} rows with missing data")
        
        # Sort by date
        if 'date' in df.columns and not df['date'].isna().all():
            df = df.sort_values('date').reset_index(drop=True)
            st.success("✅ Sorted transactions by date")
        
        st.success(f"🎉 Data cleaning complete! Final dataset: {len(df)} rows")
        return df

    def auto_categorize(self, df: pd.DataFrame, use_ai: bool = True) -> pd.DataFrame:
        """Categorize transactions using GPT or fallback rules."""
        if df.empty:
            return df
        
        df['category'] = 'Other'
        
        if use_ai and self.openai_client:
            try:
                df = self._categorize_with_gpt(df)
            except Exception as e:
                st.warning(f"AI categorization failed: {str(e)}")
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
                transactions_text.append(f"Description: {row['description']}, Amount: ${row['amount']:.2f}")
            
            prompt = f"""Categorize these transactions into: {', '.join(self.categories)}
            
            Transactions:
            {chr(10).join(transactions_text)}
            
            Return only comma-separated categories in order."""
            
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
            'Income': ['salary', 'paycheck', 'deposit', 'bonus'],
            'Groceries': ['grocery', 'supermarket', 'food', 'safeway', 'kroger'],
            'Dining & Coffee': ['restaurant', 'cafe', 'coffee', 'starbucks'],
            'Transportation': ['uber', 'lyft', 'gas', 'fuel'],
            'Shopping': ['amazon', 'target', 'walmart'],
            'Investments': self.investment_keywords
        }
        
        for category, keywords in category_rules.items():
            mask = df['description'].str.lower().str.contains('|'.join(keywords), case=False, na=False)
            df.loc[mask, 'category'] = category
        
        return df

    def create_plotly_charts(self, df: pd.DataFrame, investment_progress: Dict) -> Dict:
        """Create interactive Plotly charts with KonMari colors."""
        charts = {}
        
        if df.empty:
            return charts
        
        colors = [self.colors['soft_sage'], self.colors['blush_rose'], self.colors['muted_clay']]
        
        # Income vs Expenses
        income_data = df[df['amount'] > 0].groupby(df['date'].dt.to_period('M'))['amount'].sum()
        expense_data = df[df['amount'] < 0].groupby(df['date'].dt.to_period('M'))['amount'].sum().abs()
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=income_data.index.to_timestamp(),
            y=income_data.values,
            mode='lines+markers',
            name='Income',
            line=dict(color=self.colors['soft_sage'], width=4)
        ))
        fig1.add_trace(go.Scatter(
            x=expense_data.index.to_timestamp(),
            y=expense_data.values,
            mode='lines+markers',
            name='Expenses',
            line=dict(color=self.colors['blush_rose'], width=4)
        ))
        
        fig1.update_layout(
            title="💫 Income & Expense Flow",
            plot_bgcolor=self.colors['off_white'],
            paper_bgcolor=self.colors['off_white']
        )
        charts['income_vs_expenses'] = fig1
        
        # Spending by Category
        expense_categories = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        
        fig2 = go.Figure(data=[go.Pie(
            labels=expense_categories.index,
            values=expense_categories.values,
            marker_colors=colors
        )])
        
        fig2.update_layout(
            title="🌸 Spending by Category",
            plot_bgcolor=self.colors['off_white'],
            paper_bgcolor=self.colors['off_white']
        )
        charts['spending_by_category'] = fig2
        
        return charts

    def generate_summary(self, df: pd.DataFrame, investment_progress: Dict) -> str:
        """Generate a peaceful, natural-language summary."""
        if df.empty:
            return "No transaction data available."
        
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
        net_flow = total_income - total_expenses
        
        flow_sentiment = "You spent less than you earned" if net_flow > 0 else "Your expenses exceeded income"
        
        summary = f"""
        ### 🌱 Your Financial Reflection
        
        {flow_sentiment} this period, creating a {'positive' if net_flow > 0 else 'mindful'} 
        cash flow of **${abs(net_flow):,.2f}**.
        """
        
        return summary.strip()

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="peaceful-header">🌱 KonMari Budget</h1>', unsafe_allow_html=True)
    st.markdown('<p class="peaceful-subheader">Transform your financial data into mindful insights</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🌸 Getting Started")
        st.markdown("1. Upload your bank files\n2. Review the analysis\n3. Get peaceful insights")
        
        st.markdown("### 📁 Supported Files")
        st.markdown("• CSV files\n• Excel files\n• PDF statements")
    
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
        use_ai = st.checkbox("Use AI for categorization", value=True)
    with col2:
        send_to_notion = st.checkbox("Send results to Notion", value=True)
    
    # Process files when uploaded
    if uploaded_files:
        with st.spinner("🌱 Loading your financial data..."):
            df = dashboard.load_uploaded_files(uploaded_files)
        
        if not df.empty:
            st.success(f"✨ Loaded {len(df)} transactions")
            
            # Check if we need manual column mapping
            required_columns = ['date', 'amount', 'description']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"🔧 Need help mapping: {', '.join(missing_columns)}")
                
                with st.expander("🎯 Manual Column Mapping"):
                    st.write("Available columns:", df.columns.tolist())
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'date' not in df.columns:
                            date_col = st.selectbox("Select DATE column:", [''] + df.columns.tolist())
                            if date_col:
                                df = df.rename(columns={date_col: 'date'})
                    
                    with col2:
                        if 'amount' not in df.columns:
                            amount_col = st.selectbox("Select AMOUNT column:", [''] + df.columns.tolist())
                            if amount_col:
                                df = df.rename(columns={amount_col: 'amount'})
                    
                    with col3:
                        if 'description' not in df.columns:
                            desc_col = st.selectbox("Select DESCRIPTION column:", [''] + df.columns.tolist())
                            if desc_col:
                                df = df.rename(columns={desc_col: 'description'})
            
            # Show data preview
            with st.expander("👀 Preview Your Data"):
                st.dataframe(df.head(10))
            
            # Run analysis button
            if all(col in df.columns for col in required_columns):
                if st.button("🚀 Create My Financial Reflection", type="primary"):
                    
                    # Categorize
                    df = dashboard.auto_categorize(df, use_ai=use_ai)
                    
                    # Generate summary
                    investment_progress = {}
                    summary = dashboard.generate_summary(df, investment_progress)
                    
                    # Create charts
                    charts = dashboard.create_plotly_charts(df, investment_progress)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown(summary)
                    
                    # Key metrics
                    st.markdown("### 📊 Key Insights")
                    col1, col2, col3 = st.columns(3)
                    
                    total_income = df[df['amount'] > 0]['amount'].sum()
                    total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
                    net_flow = total_income - total_expenses
                    
                    with col1:
                        st.metric("Net Cash Flow", f"${net_flow:,.2f}")
                    with col2:
                        st.metric("Total Income", f"${total_income:,.2f}")
                    with col3:
                        st.metric("Total Expenses", f"${total_expenses:,.2f}")
                    
                    # Display charts
                    for chart_name, chart in charts.items():
                        st.plotly_chart(chart, use_container_width=True)
                    
                    # Transaction details
                    with st.expander("📋 Categorized Transactions"):
                        st.dataframe(df[['date', 'description', 'amount', 'category']])
                    
                    st.balloons()
            else:
                missing = [col for col in required_columns if col not in df.columns]
                st.error(f"❌ Missing columns: {', '.join(missing)}")
        else:
            st.error("No valid transaction data found.")
    
    else:
        # Welcome message
        st.markdown("""
        ### Welcome to your peaceful financial journey! 🌱
        
        Upload your bank statements to get started with mindful financial insights.
        """)

if __name__ == "__main__":
    main()
