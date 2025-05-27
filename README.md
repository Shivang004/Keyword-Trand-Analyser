## Deployment

Link: https://keyword-trend-analyser.streamlit.app/

# Research Analytics Dashboard - How to Use Guide

## 📋 Overview

The Research Analytics Dashboard is a comprehensive tool for exploring research trends across years, countries, and keywords. It provides interactive visualizations and semantic search capabilities to help you discover patterns in research data.

## 🎛️ Getting Started

### Summary Granularity Selection
Before diving into the analysis, choose your preferred data granularity from the sidebar:
- **Fine-grained (~9000)**: Most detailed analysis with approximately 9,000 data points
- **Moderate (~4000)**: Balanced view with around 4,000 data points  
- **Broad (~1800)**: High-level overview with about 1,800 data points

> 💡 **Tip**: Start with "Moderate" for a good balance between detail and performance.

### Navigation
Use the sidebar navigation to jump between different analysis sections. Each section offers unique insights into the research data.

## 📊 Analysis Sections

### 1. Year → Keywords
**Purpose**: Discover the most popular research keywords in a specific year.

**How to use**:
1. Select a year from the dropdown
2. Adjust the slider to show top N keywords (5-30)
3. Optionally filter to specific keywords using the multiselect
4. View results in both bar chart and data table formats

**Best for**: Understanding what topics were trending in a particular year.

### 2. Year → Countries
**Purpose**: Identify the most research-active countries in a selected year.

**How to use**:
1. Choose a year (note: 2018 country data is missing)
2. Set the number of top countries to display
3. Filter to specific countries if needed
4. Explore both bar chart and world map visualizations

**Best for**: Geographic analysis of research activity by year.

### 3. Keyword → Years
**Purpose**: Track how keyword popularity changes over time.

**How to use**:
1. Select one or multiple keywords from the list
2. Compare trends using the line chart
3. View year-wise distribution in the bar chart
4. Analyze the data table for exact values

**Best for**: Longitudinal analysis of research topic trends.

### 4. Keyword → Countries
**Purpose**: See which countries are most active in researching specific keywords.

**How to use**:
1. Choose keywords of interest
2. Set the number of top countries to display
3. View comparative bar charts for multiple keywords
4. Explore the choropleth map for geographic insights

**Best for**: Understanding global research distribution by topic.

### 5. Country → Years
**Purpose**: Analyze a country's research activity over time.

**How to use**:
1. Select countries to compare (note: 2018 data missing)
2. View trends in line chart format
3. Compare year-wise distributions
4. Export data table for further analysis

**Best for**: Tracking national research output trends.

### 6. Country → Keywords
**Purpose**: Discover what topics countries focus on most.

**How to use**:
1. Select countries for comparison
2. Adjust the number of top keywords shown
3. Compare research focus areas across countries
4. Use the data table for detailed analysis

**Best for**: Understanding national research priorities.

### 7. Year + Keyword → Countries
**Purpose**: Find which countries researched specific keywords in a particular year.

**How to use**:
1. Choose a target year
2. Select keywords of interest
3. Set the number of countries to display
4. View both bar chart and world map visualizations

**Best for**: Year-specific geographic analysis of research topics.

### 8. Year + Country → Keywords
**Purpose**: Understand what specific countries researched in a given year.

**How to use**:
1. Select the year of interest
2. Choose countries to analyze
3. Adjust the number of top keywords displayed
4. Compare research focus across selected countries

**Best for**: Analyzing national research agendas for specific years.

### 9. Keyword + Country → Years
**Purpose**: Track when countries were most active in specific research areas.

**How to use**:
1. Select keywords and countries of interest
2. View temporal trends in line charts
3. Analyze year-wise activity patterns
4. Compare multiple keyword-country combinations

**Best for**: Understanding the timing of research focus by country and topic.

## 🔍 Semantic Search

**Purpose**: Find keywords related to your research interests using AI-powered similarity matching.

**How to use**:
1. Enter a keyword or phrase in the search box
2. Review the top 20 most similar keywords with similarity scores
3. Select relevant keywords from the results
4. Automatically generate three types of analysis:
   - **Trend Over Time**: See how selected keywords evolved yearly
   - **Countries Research**: Discover which countries research these topics
   - **Keyword + Country → Years**: Track specific country activity over time

**Advanced features**:
- Uses sentence transformers for intelligent matching
- Automatically adapts to your selected granularity setting
- Provides similarity scores for transparency
- Enables multi-keyword comparative analysis

## 📈 Chart Types and Interpretations

### Bar Charts
- **Use**: Comparing quantities across categories
- **Interpretation**: Longer bars indicate higher values
- **Interactive**: Hover for exact values

### Line Charts  
- **Use**: Showing trends over time
- **Interpretation**: Slopes indicate growth/decline rates
- **Interactive**: Multiple lines allow comparison

### Choropleth Maps
- **Use**: Geographic data visualization
- **Interpretation**: Darker colors indicate higher activity
- **Interactive**: Click countries for details

### Data Tables
- **Use**: Exact numerical values
- **Features**: Sortable columns, full data export capability

## ⚠️ Important Notes

### Data Limitations
- **2019 Data**: Not available and excluded from all charts
- **2018 Country Data**: Missing from country-related analyses
- **Granularity Impact**: Different settings may show different trends

### Performance Tips
- Start with "Moderate" granularity for best performance
- Select fewer items when comparing multiple keywords/countries
- Use semantic search to discover relevant keywords efficiently

### Best Practices
- **Compare Related Items**: Select multiple similar keywords or countries for meaningful comparisons
- **Use Multiple Views**: Combine different analysis sections for comprehensive insights
- **Leverage Semantic Search**: Don't limit yourself to exact keyword matches
- **Export Data**: Use data tables to export findings for further analysis

## 🎯 Common Use Cases

### Academic Researchers
- Track research trends in your field over time
- Identify emerging research areas
- Find international collaboration opportunities
- Benchmark your country's research activity

### Policy Makers
- Understand national research priorities
- Compare research output with other countries
- Identify gaps in research focus areas
- Track the evolution of policy-relevant topics

### Research Administrators
- Monitor institutional research alignment with global trends
- Identify potential collaboration partners by geography and topic
- Track the impact of research funding decisions over time
- Plan future research investments based on trending topics

### Students and Early Career Researchers
- Discover hot research topics in your field
- Identify countries leading in specific research areas
- Find potential mentors or collaboration opportunities
- Understand how research interests have evolved over time

## 🚀 Getting the Most Out of the Dashboard

1. **Start Broad**: Begin with overview sections (1, 2, 5) to understand general patterns
2. **Drill Down**: Use specific combinations (7, 8, 9) for targeted insights
3. **Use Semantic Search**: Discover related topics you might have missed
4. **Compare Multiple Items**: Always select multiple keywords/countries for richer analysis
5. **Cross-Reference**: Use findings from one section to inform exploration in others
6. **Export Data**: Save interesting findings using the data tables for further analysis

---

*Dashboard created with ❤️ using Streamlit | Shivang Agarwal*
