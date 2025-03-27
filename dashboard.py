import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Load Data with Caching
@st.cache_resource
def load_data():
    file_path = "DataAnalyst_Assesment_Dataset.xlsx"
    df = pd.read_excel(file_path)
    df['Booking Date'] = pd.to_datetime(df['Booking Date'], errors='coerce')
    return df


def Sidebar_Filters(df):
    st.sidebar.header("Filters")
    min_date, max_date = df['Booking Date'].min(), df['Booking Date'].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    service_type = st.sidebar.multiselect("Select Service Type", options=df['Service Type'].dropna().unique(), default=df['Service Type'].dropna().unique())
    return date_range, service_type

def Filter_Data(df, date_range, service_type):
    filtered_df = df[(df['Booking Date'] >= pd.to_datetime(date_range[0])) & (df['Booking Date'] <= pd.to_datetime(date_range[1]))]
    filtered_df = filtered_df[filtered_df['Service Type'].isin(service_type)]
    filtered_df = filtered_df.dropna(subset=['Price', 'Customer ID'])  # Handle missing values
    return filtered_df

def KPIs(filtered_df):
    st.title("📊 Booking Analysis Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bookings", len(filtered_df))
    col2.metric("Confirmed Bookings", len(filtered_df[filtered_df["Status"] == "Confirmed"]))
    col3.metric("Avg. Booking Price ($)", f"{filtered_df['Price'].mean():,.2f}")
    col4.metric("Most Popular Service Type", filtered_df['Service Name'].mode()[0] if not filtered_df['Service Name'].empty else "N/A")

def Charts(filtered_df):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Booking Status Distribution")
        status_counts = filtered_df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig1 = px.pie(status_counts, names='Status', values='Count', title="Booking Status Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Revenue by Service Type")
        service_revenue = filtered_df.groupby('Service Type')['Price'].sum().reset_index()
        fig2 = px.bar(service_revenue, x='Service Type', y='Price', title='Revenue by Service Type', text_auto=True)
        st.plotly_chart(fig2, use_container_width=True)

def Trend(filtered_df):
    st.subheader("Revenue Over Time")
    filtered_df['Booking Date'] = pd.to_datetime(filtered_df['Booking Date'])
    daily_revenue = filtered_df.groupby(filtered_df['Booking Date'].dt.date)['Price'].sum().reset_index()
    fig2 = px.line(daily_revenue, x='Booking Date', y='Price', title='Daily Revenue Trend', markers=True)
    st.plotly_chart(fig2, use_container_width=True)
    
def Booking(filtered_df):
    st.subheader("Confirmed Status with Missing Data")
    
    # Filter only confirmed bookings
    confirmed_df = filtered_df[filtered_df["Status"] == "Confirmed"]
    
    # Identifying missing values
    missing_data = confirmed_df.isnull().sum() + confirmed_df.eq(" ").sum()
    missing_data = missing_data[missing_data > 0]  # Keep only columns with missing values
    missing_data = missing_data.reset_index()
    missing_data.columns = ["Column", "Missing Values"]

    # Extract relevant data for download
    pending_data_with_ids = confirmed_df[["Booking ID", "Customer ID", "Customer Name"]].copy()
    
    # Merge missing data summary with booking details
    if not missing_data.empty:
        missing_data_report = pd.DataFrame(missing_data)
        pending_data_with_ids["Missing Data Summary"] = "Check MissingData.csv"

        # Plot missing values
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=missing_data["Column"], y=missing_data["Missing Values"], palette="viridis", ax=ax)
        plt.xticks(rotation=90)
        plt.xlabel("Columns")
        plt.ylabel("Number of Missing Values")
        plt.title("Missing Data in Columns")
        st.pyplot(fig)

        # Save data for download
        csv_data = pending_data_with_ids.to_csv(index=False).encode("utf-8")
        csv_missing_data = missing_data_report.to_csv(index=False).encode("utf-8")

        col1, col2 = st.columns(2)
        
        with col1:
            # Provide a download button for booking data
            st.download_button(
                label="Download Booking Data",
                data=csv_data,
                file_name="ConfirmedBookingData.csv",
                mime="text/csv",
                help="Click here to download the booking details with missing data reference",
            )
        with col2:    
            # Provide a download button for missing data report
            st.download_button(
                label="Download Missing Data Report",
                data=csv_missing_data,
                file_name="ConfirmedMissingData.csv",
                mime="text/csv",
                help="Click here to download the missing data report",
            )

def Heatmap(df):
    st.subheader("Feature Correlation Heatmap")
    selected_columns = ["Booking Type", "Class Type", "Duration (mins)", "Facility", "Instructor", "Price", "Service Name", "Service Type", "Status", "Theme"]
    df_selected = df[selected_columns]
    
    for col in df_selected.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_selected[col] = le.fit_transform(df_selected[col].astype(str))
    
    corr_matrix_selected = df_selected.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix_selected, annot=True, cmap="coolwarm", center=0, linewidths=0.5, fmt=".2f")
    st.pyplot(plt)



def Filtered_Data(filtered_df):
    st.subheader("Cleaned Filtered Data")
    st.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.filtered_data = filtered_df
        # Button to navigate to dashboard
        if st.button("Go to Dashboard"):
            st.switch_page("pages\dashboardOfFilterData.py")
    
    with col2:
        st.download_button("Download Data", data=csv, file_name="Filtered_Data.csv", mime="text/csv")


def handlingmissing(df):
    
    # Handling Missing Values
    df['Class Type'].fillna("Unknown", inplace=True)
    df['Instructor'].fillna("Not Assigned", inplace=True)
    df['Facility'].fillna("Unknown", inplace=True)
    df['Theme'].fillna("Not Specified", inplace=True)
    df['Time Slot'].fillna(df['Time Slot'].mode()[0], inplace=True)
    df['Duration (mins)'].fillna(df['Duration (mins)'].median(), inplace=True)
    df['Price'].fillna(df['Price'].median(), inplace=True)
    df['Customer Email'].fillna("Not Specified", inplace = True)
    df['Customer Phone'].fillna("Not Sepcified", inplace = True)
    if 'Subscription Type' in df.columns:
        df.drop(columns=['Subscription Type'], inplace=True)
    if 'Status' in df.columns:
        df = df[df['Status'] != 'Pending']
    return df


def main():
    df = load_data()
    date_range, service_type = Sidebar_Filters(df)
    filtered_df = Filter_Data(df, date_range, service_type)
    
    st.markdown("<br><br>", unsafe_allow_html=True)  # Bigger gap
    KPIs(filtered_df)
    st.markdown("<br><br>", unsafe_allow_html=True)  # Bigger gap
    Charts(filtered_df)
    st.markdown("<br><br>", unsafe_allow_html=True)  # Bigger gap
    Trend(filtered_df)
    st.markdown("<br><br>", unsafe_allow_html=True)  # Bigger gap
    Booking(filtered_df)
    st.markdown("<br><br>", unsafe_allow_html=True)  # Bigger gap
    Heatmap(filtered_df)
    st.markdown("<br><br>", unsafe_allow_html=True)  # Bigger gap
    cleaneddata = handlingmissing(filtered_df)
    st.markdown("<br><br>", unsafe_allow_html=True)  # Bigger gap
    Filtered_Data(cleaneddata)
    


if __name__ == "__main__":
    main()