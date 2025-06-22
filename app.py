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
    page_title="🌱 Peaceful Finance Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for peaceful styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stTitle {
        color: #9CAF88;
        font-family: 'Helvetica', sans-serif;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #9CAF88;
        background-color: #F9F7F4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F9F7F4;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E1DA;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PeacefulFinanceDashboard:
    """Streamlit version of the Peaceful Finance Dashboard"""
    
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
            'sage_green': '#9CAF88',
            'warm_sand': '#F4E4C1',
            'blush_rose': '#E8B4B8',
            'clay': '#D4A574',
            'off_white': '#F9F7F4',
            'pale_blue': '#B8D4E3',
            'soft_gray': '#E5E1DA'
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
        """Extract transaction data from PDF bank statements."""
        transactions = []
        
        try:
            # Try pdfplumber first
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 1:
                            headers = table[0]
                            for row in table[1:]:
                                if len(row) >= 3:
                                    transactions.append(dict(zip(headers, row)))
        except:
            # Fallback to PyMuPDF
            try:
                doc = fitz.open(file_path)
                for page in doc:
                    tables = page.find_tables()
                    for table in tables:
                        df = table.to_pandas()
                        if not df.empty:
                            transactions.extend(df.to_dict('records'))
                doc.close()
            except:
                st.warning(f"Could not parse PDF: {file_path}")
        
        return pd.DataFrame(transactions) if transactions else pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and detect date, description, amount columns."""
        df.columns = df.columns.str.lower().str.strip()
        
        # Common column mappings
        column_mapping = {
            'transaction date': 'date',
            'trans date': 'date',
            'posting date': 'date',
            'transaction description': 'description',
            'trans description': 'description',
            'merchant': 'description',
            'memo': 'description',
            'debit': 'amount',
            'credit': 'amount',
            'transaction amount': 'amount',
            'trans amount': 'amount'
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
        
        if 'description' not in df.columns:
            desc_cols = [col for col in df.columns if any(word in col.lower() 
                        for word in ['desc', 'merchant', 'memo', 'payee'])]
            if desc_cols:
                df = df.rename(columns={desc_cols[0]: 'description'})
        
        if 'amount' not in df.columns:
            amount_cols = [col for col in df.columns if any(word in col.lower() 
                          for word in ['amount', 'debit', 'credit', 'value'])]
            if amount_cols:
                df = df.rename(columns={amount_cols[0]: 'amount'})
        
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the transaction data."""
        if df.empty:
            return df
        
        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean amounts
        if 'amount' in df.columns:
            df['amount'] = df['amount'].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Clean descriptions
        if 'description' in df.columns:
            df['description'] = df['description'].astype(str).str.strip()
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['date', 'amount', 'description'])
        
        # Sort by date
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
        """Create interactive Plotly charts."""
        charts = {}
        
        if df.empty:
            return charts
        
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
            line=dict(color='#9CAF88', width=3),
            marker=dict(size=8)
        ))
        fig1.add_trace(go.Scatter(
            x=expense_data.index.to_timestamp(),
            y=expense_data.values,
            mode='lines+markers',
            name='Expenses',
            line=dict(color='#E8B4B8', width=3),
            marker=dict(size=8)
        ))
        
        fig1.update_layout(
            title="💫 Income & Expense Flow",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            plot_bgcolor='#F9F7F4',
            paper_bgcolor='#F9F7F4',
            font=dict(color='#5A5A5A'),
            showlegend=True
        )
        charts['income_vs_expenses'] = fig1
        
        # 2. Spending by Category
        expense_categories = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        
        fig2 = go.Figure(data=[go.Pie(
            labels=expense_categories.index,
            values=expense_categories.values,
            marker_colors=['#9CAF88', '#E8B4B8', '#D4A574', '#B8D4E3', '#F4E4C1', '#E5E1DA']
        )])
        
        fig2.update_layout(
            title="🌸 Spending Reflection by Category",
            plot_bgcolor='#F9F7F4',
            paper_bgcolor='#F9F7F4',
            font=dict(color='#5A5A5A')
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
                marker_color='#9CAF88',
                text=[f'{pct:.1f}%' for pct in progress_pcts],
                textposition='outside'
            )])
            
            fig3.update_layout(
                title="🎯 Investment Goal Journey",
                xaxis_title="Progress (%)",
                plot_bgcolor='#F9F7F4',
                paper_bgcolor='#F9F7F4',
                font=dict(color='#5A5A5A'),
                xaxis=dict(range=[0, 100])
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
    
    # Header
    st.title("🌱 Peaceful Finance Dashboard")
    st.markdown("*Transform your financial data into mindful insights*")
    
    # Sidebar
    with st.sidebar:
        st.header("🌸 Getting Started")
        st.markdown("""
        1. Upload your bank/credit card files
        2. Choose your analysis preferences  
        3. Let the magic happen ✨
        4. Review your peaceful financial reflection
        """)
        
        st.header("📁 Supported Files")
        st.markdown("• CSV files (.csv)")
        st.markdown("• Excel files (.xlsx, .xls)")
        st.markdown("• PDF statements (.pdf)")
        
        # API Status
        st.header("🔧 Configuration")
        openai_status = "✅" if st.secrets.get("OPENAI_API_KEY") else "❌"
        notion_status = "✅" if st.secrets.get("NOTION_TOKEN") else "❌"
        
        st.markdown(f"OpenAI API: {openai_status}")
        st.markdown(f"Notion API: {notion_status}")
    
    # Initialize dashboard
    dashboard = PeacefulFinanceDashboard()
    
    # File upload
    st.header("📁 Upload Your Financial Data")
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
            
            # Show data preview
            with st.expander("👀 Preview Your Data"):
                st.dataframe(df.head(10))
            
            # Run analysis button
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
            st.error("No valid transaction data found in uploaded files. Please check your file format.")
    
    else:
        # Welcome message
        st.markdown("""
        ### Welcome to your peaceful financial journey! 🌱
        
        Upload your bank statements, credit card files, or investment PDFs to get started. 
        This tool will mindfully analyze your spending patterns and help you track progress 
        toward your financial goals with beautiful, calming visualizations.
        
        **What you'll get:**
        - 🏷️ Intelligent transaction categorization
        - 📊 Beautiful spending insights
        - 🎯 Investment goal tracking
        - 📝 Peaceful financial summary
        - 🚀 Automatic Notion integration
        """)

if __name__ == "__main__":
    main()