"""
Name:       Naomi Vo
CS230:      Section 6
Data:       Nuclear Explosions
URL:        Link to your web application on Streamlit Cloud (if posted)
Description:
This program analyzes data on nuclear explosions conducted between 1945 and 1998 and tell a story about nuclear
explosions with real world data. It includes features to filter and visualize the dataset based on specific criteria,
such as country, year range, and test type. The program visualizes the data through various charts, including bar chart
that displays the highest yield nuclear tests. Pie chart shows the distribution of nuclear tests by their purpose.
Line chart visualizes the trend of nuclear tests over time. Scatter plot maps the locations of nuclear tests. An
interactive map is embedded into the application, where users can view the locations of the nuclear tests with clickable
markers. Each marker provides additional information about the test, such as the name and yield, offering a more
immersive experience. In addition to the visualizations, users can explore the filtered dataset in a table format,
allowing for a deeper dive into the raw data. This option is helpful for users who wish to conduct more detailed analysis.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

# Constants
DEFAULT_YEAR_RANGE = (1945, 1998)  # [PY1] Default value for the year range
PIE_CHART_THRESHOLD = 0.05

# Load and read data
def load_data(file_path):
    # [PY3] Error checking with try/except
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure the file is in the correct directory.")
        st.stop()

# [DA1] Clean data
def clean_data(df):
    df.dropna(subset=['Date.Year', 'Date.Month', 'Date.Day'], inplace=True)
    df = df[(1 <= df['Date.Month']) & (df['Date.Month'] <= 12)]
    df = df[(1 <= df['Date.Day']) & (df['Date.Day'] <= 31)]
    df['Date'] = pd.to_datetime(df['Date.Year'].astype(str) + '-' +
                                df['Date.Month'].astype(str) + '-' +
                                df['Date.Day'].astype(str), errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df['Year'] = df['Date'].dt.year
    return df

# [PY1] Function with parameters (one with default value)
def get_top_yield_tests(df, top_n=5):
    # Get top nuclear tests by yield
    return df.nlargest(top_n, 'Data.Yeild.Upper')[['Data.Name', 'Data.Yeild.Upper']]

# [PY2] Function that returns multiple values
def summarize_tests_by_country(df):
    # Summarize test counts by country
    country_counts = {country: count for country, count in df['WEAPON SOURCE COUNTRY'].value_counts().items()}
    return country_counts    # [PY5] Dictionary of counts

# Visualization Functions
# [VIZ1] Bar chart with titles, labels, and colors
def plot_bar_chart(df):
    # Bar chart for highest yield nuclear tests
    top_tests = get_top_yield_tests(df, top_n=10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_tests['Data.Name'], top_tests['Data.Yeild.Upper'], color='purple')
    ax.set(title="Highest Yield Nuclear Tests", xlabel="Yield (Kiloton)", ylabel="Test Name")
    st.pyplot(fig)

# Explanation below the chart
    st.markdown("""
    **Highest Yield Nuclear Tests**:  
    This bar chart displays the top nuclear tests by their yield (in kilotons). Each bar represents the yield of a 
    specific test, with the test names listed on the y-axis. The chart helps to identify which tests had the highest 
    destructive potential. To adjust the number of tests shown, simply change the filter settings.
    """)

# [VIZ2] Pie chart with labels and titles
def plot_pie_chart(df):
    # Pie chart for nuclear tests by purpose
    purpose_counts = df['Data.Purpose'].value_counts()
    small_categories = purpose_counts[purpose_counts < PIE_CHART_THRESHOLD * purpose_counts.sum()].index
    purpose_counts_adjusted = purpose_counts.copy()
    purpose_counts_adjusted.loc["Other"] = purpose_counts_adjusted.loc[small_categories].sum()
    purpose_counts_adjusted = purpose_counts_adjusted.drop(small_categories)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(purpose_counts_adjusted, labels=purpose_counts_adjusted.index, autopct='%1.0f%%', startangle=90)
    ax.set_title("Nuclear Tests by Purpose")
    st.pyplot(fig)

# Explanation below the chart
    st.markdown("""
    **Distribution of Nuclear Tests by Purpose**:  
    The pie chart visualizes the distribution of nuclear tests based on their stated purpose such as military, peaceful, 
    or scientific purposes. Each slice of the pie represents the proportion of tests categorized by purpose. If any 
    category has a small proportion, it is grouped into the "Other" category to simplify the view.
    """)

# [VIZ1] Line chart with labels and titles
def plot_line_chart(df):
    # Line chart for nuclear tests over time
    tests_per_year = df.groupby('Year').size()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tests_per_year.index, tests_per_year.values, marker="o", color="b")
    ax.set(title="Nuclear Tests Over Time", xlabel="Year", ylabel="Number of Tests")
    st.pyplot(fig)

# Explanation below the chart
    st.markdown("""
    **Trends in Nuclear Tests Over Time**:  
    This line chart shows how the number of nuclear tests changed from year to year, highlighting trends in testing 
    frequency. The x-axis represents the years, while the y-axis shows the number of tests conducted each year. 
    This visualization helps to identify periods of increased or decreased testing activity. Observe the peaks and 
    troughs in testing activity, which could correspond to political events or international treaties.
    """)

# [VIZ2] Scatter plot with test locations (map visualization)
def plot_scatter_map(df):
    # Scatter plot for nuclear test locations
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Location.Cordinates.Longitude'], df['Location.Cordinates.Latitude'], c='red', marker='o')
    ax.set(title="Test Locations", xlabel="Longitude", ylabel="Latitude")
    st.pyplot(fig)

# Explanation below the chart
    st.markdown("""
    **Locations of Nuclear Tests**:  
    The scatter plot shows the geographical locations of nuclear tests on a global scale. Each point represents a test, 
    with longitude on the x-axis and latitude on the y-axis. This chart provides a clear picture of where nuclear tests 
    were predominantly conducted. Use this scatter plot to quickly see which regions were most active in nuclear testing.
    """)

# [MAP] Detailed map with markers, text and tooltips
def display_map(df):
    # Interactive map for test locations
    # Initialize the map centered on the mean latitude and longitude
    test_map = folium.Map(
        location=[df['Location.Cordinates.Latitude'].mean(), df['Location.Cordinates.Longitude'].mean()],
        zoom_start=3
    )

    # [DA8] Iterate through DataFrame with iterrows()
    for index, row in df.iterrows():
        # Add CircleMarkers instead of default Markers
        folium.CircleMarker(
            location=[row['Location.Cordinates.Latitude'], row['Location.Cordinates.Longitude']],
            radius=5,  # Adjust size of the circle
            color='blue',  # Circle border color
            fill=True,
            fill_color='blue',  # Circle fill color
            fill_opacity=0.7,  # Opacity of the fill
            popup=f"Test: {row['Data.Name']}<br>Yield: {row['Data.Yeild.Upper']} kt" if not pd.isnull(row['Data.Name']) else None,
            tooltip=row['Data.Name'] if not pd.isnull(row['Data.Name']) else None,
        ).add_to(test_map)

    # Display the map
    st_folium(test_map, width=700, height=500)

# Explanation below the map
    st.markdown("""
    **Interactive Map with Test Locations**:  
    The interactive map allows you to zoom in and explore the locations of individual nuclear tests. Each test is 
    marked with a clickable circle, providing detailed information about the test, such as its name and yield. Hover 
    over any of the markers to view more details, and click to get the specific test’s data.
    """)


# Main Application
st.set_page_config(page_title="Nuclear Explosions Analysis", layout="wide")
st.title("Nuclear Explosions Analysis: 1945-1998")
st.sidebar.header("Filters")

# Load and prepare data
df = load_data("nuclear_explosions.csv")
df = clean_data(df)

# Sidebar filters
countries = summarize_tests_by_country(df)  # [DA2] Summarize data by country
# [ST1] Sidebar dropdown
selected_country = st.sidebar.selectbox("Select a Country", options=["All"] + list(countries.keys()), index=0)
# [ST2] Sidebar slider
year_range = st.sidebar.slider("Select Year Range", int(df['Year'].min()), int(df['Year'].max()), DEFAULT_YEAR_RANGE)
# [ST3] Sidebar multi-selects
test_types = st.sidebar.multiselect("Select Test Types", options=df['Data.Type'].unique(), default=df['Data.Type'].unique())

# Apply filters
if selected_country != "All":
    df = df[df['WEAPON SOURCE COUNTRY'] == selected_country]
df = df[df['Year'].between(year_range[0], year_range[1])]
df = df[df['Data.Type'].isin(test_types)]

# Display filtered data
st.subheader("Filtered Dataset")
st.dataframe(df)  # [DA9] Display filtered data in a table


# Visualizations
st.subheader("Visualizations")
plot_bar_chart(df)
plot_pie_chart(df)
plot_line_chart(df)
plot_scatter_map(df)
display_map(df)
