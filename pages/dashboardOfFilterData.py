import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Check if filtered data exists
if "filtered_data" in st.session_state and not st.session_state.filtered_data.empty:
    filtered_df = st.session_state.filtered_data
    
# Convert Booking Date to datetime
filtered_df['Booking Date'] = pd.to_datetime(filtered_df['Booking Date'])

# Display Data
st.write("### Dataset Overview")
st.dataframe(filtered_df.head())

# Booking Type Distribution using a Pie Chart
st.write("### Booking Type Distribution")
booking_counts = filtered_df['Booking Type'].value_counts().reset_index()
booking_counts.columns = ['Booking Type', 'Count']
fig_booking_type = px.pie(booking_counts, names='Booking Type', values='Count', title='Booking Type Distribution')
st.plotly_chart(fig_booking_type)

# Revenue by Service Type using a Bar Chart
st.write("### Revenue by Service Type")
revenue_service = filtered_df.groupby('Service Type')['Price'].sum().reset_index()
fig_revenue = px.bar(revenue_service, x='Service Type', y='Price', title='Revenue Breakdown', labels={'Price': 'Total Revenue'})
st.plotly_chart(fig_revenue)

# Peak Booking Dates using an Area Chart
st.write("### Peak Booking Dates")
bookings_per_date = filtered_df.groupby(filtered_df['Booking Date'].dt.date)['Booking Type'].count().reset_index()
bookings_per_date.columns = ['Booking Date', 'Count']
fig_bookings = px.area(bookings_per_date, x='Booking Date', y='Count', title='Bookings Over Time', labels={'Count': 'Total Bookings'})
st.plotly_chart(fig_bookings)

# Instructor Workload using a Horizontal Bar Chart
st.write("### Instructor Workload")
filtered_df_instructors = filtered_df[filtered_df['Instructor'] != "Not Assigned"]  # Exclude placeholders
instructor_counts = filtered_df_instructors['Instructor'].value_counts().reset_index()
instructor_counts.columns = ['Instructor', 'Bookings Handled']
fig_instructors = px.bar(instructor_counts, x='Bookings Handled', y='Instructor', orientation='h', title='Instructor Workload')
st.plotly_chart(fig_instructors)

# Revenue Trends Over Time using a Line Chart
st.write("### Revenue Trends Over Time")
revenue_trend = filtered_df.groupby(filtered_df['Booking Date'].dt.date)['Price'].sum().reset_index()
fig_revenue_trend = px.line(revenue_trend, x='Booking Date', y='Price', title='Revenue Trends', labels={'Price': 'Total Revenue'})
st.plotly_chart(fig_revenue_trend)

# Customer Booking Patterns by Time Slot using a Bar Chart
st.write("### Customer Booking Patterns by Time Slot")
time_slot_counts = filtered_df['Time Slot'].value_counts().reset_index()
time_slot_counts.columns = ['Time Slot', 'Count']
fig_time_slot = px.bar(time_slot_counts, x='Time Slot', y='Count', title='Bookings by Time Slot')
st.plotly_chart(fig_time_slot)

# Facility Usage Distribution using a Pie Chart
st.write("### Facility Usage Distribution")
facility_counts = filtered_df['Facility'].value_counts().reset_index()
facility_counts.columns = ['Facility', 'Count']
fig_facility = px.pie(facility_counts, names='Facility', values='Count', title='Facility Usage Distribution')
st.plotly_chart(fig_facility)

# Correlation Between Duration & Price using a Scatter Plot
st.write("### Correlation Between Duration and Price")
fig_duration_price = px.scatter(filtered_df, x='Duration (mins)', y='Price', title='Duration vs. Price', labels={'Duration (mins)': 'Duration (minutes)', 'Price': 'Price'})
st.plotly_chart(fig_duration_price)

# Service Type Popularity by Month using a Heatmap
st.write("### Service Type Popularity by Month")
filtered_df['Month'] = filtered_df['Booking Date'].dt.month
service_month_counts = filtered_df.groupby(['Month', 'Service Type']).size().reset_index(name='Count')
fig_service_heatmap = px.density_heatmap(service_month_counts, x='Month', y='Service Type', z='Count', title='Service Type Popularity by Month')
st.plotly_chart(fig_service_heatmap)

st.session_state.filtered_data = filtered_df
    # Button to navigate to dashboard
if st.button("Go to Dashboard"):
    st.switch_page("pages/PricePrediction.py")
