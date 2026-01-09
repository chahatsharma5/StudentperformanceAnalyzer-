import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ==========================================
# Data Loading and Preprocessing
# ==========================================
def load_data():
    df = pd.read_csv("student-por.csv", sep=',')

    # Normalize column names
    df.columns = [col.lower() for col in df.columns]

    required_cols = ['studytime', 'absences', 'g1', 'g2', 'g3', 'age', 'school']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. Found columns: {df.columns.tolist()}")

    # Map studytime to daily hours
    study_map = {1: 1.5, 2: 3.5, 3: 7.5, 4: 12.0}
    df['dailystudyhours'] = df['studytime'].map(study_map)

    # Attendance %
    df['attendance'] = 100 - (df['absences'] / df['absences'].max()) * 100
    df['attendance'] = df['attendance'].clip(lower=0)

    # Final result average
    df['finalresult'] = round((df['g1'] + df['g2'] + df['g3']) / 3)

    # Grade category
    df['gradecategory'] = np.where(df['finalresult'] >= 10, 'Pass', 'Fail')

    # Age groups
    df['agegroup'] = pd.cut(df['age'], bins=[14, 16, 18, 20, 22],
                            labels=['15-16', '17-18', '19-20', '21+'], right=False)
    return df


df = load_data()

# ==========================================
# K-Means Clustering
# ==========================================
features = ['dailystudyhours', 'attendance', 'finalresult', 'absences']
X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Label clusters based on performance
cluster_mean = df.groupby('cluster')['finalresult'].mean().sort_values()
label_map = {cluster_mean.index[0]: 'Low Performers',
             cluster_mean.index[1]: 'Average Performers',
             cluster_mean.index[2]: 'High Performers'}
df['cluster_label'] = df['cluster'].map(label_map)

# Cluster summaries
cluster_mean = df.groupby('cluster_label')[features].mean().round(2).reset_index()
cluster_sd = df.groupby('cluster_label')[features].std().round(2).reset_index()

# ==========================================
# Correlation Insight
# ==========================================
corr = df[features].corr()
corr_target = corr['finalresult'].drop('finalresult').sort_values(ascending=False)
best_feature = corr_target.index[0]
best_corr_value = corr_target.iloc[0]

# ==========================================
# Dash App Initialization
# ==========================================
app = Dash(__name__)
app.title = "Student Performance Dashboard (Enhanced)"

app.layout = html.Div([
    html.H1("🎓 Student Performance Analyzer Dashboard", style={'textAlign': 'center'}),

    html.Hr(),

    # ================= Filters =================
    html.Div([
        html.Div([
            html.Label("Select School:"),
            dcc.Dropdown(
                id='school-filter',
                options=[{'label': s, 'value': s} for s in sorted(df['school'].unique())],
                value=df['school'].unique()[0],
                clearable=False
            )
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Select Age Group:"),
            dcc.Dropdown(
                id='age-filter',
                options=[{'label': str(a), 'value': a} for a in df['agegroup'].dropna().unique()],
                value=df['agegroup'].dropna().unique()[0],
                clearable=False
            )
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'marginBottom': '30px'}),

    html.Div(id='auto-insight', style={'textAlign': 'center', 'fontSize': '18px', 'marginBottom': '20px'}),

    # ================= Graphs =================
    html.Div([
        dcc.Graph(id='bar-chart'),
        dcc.Graph(id='pie-chart')
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),

    html.Div([
        dcc.Graph(id='scatter-chart'),
        dcc.Graph(id='histogram')
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),

    html.Div(id='stats-output', style={'marginTop': '30px', 'padding': '10px'}),

    html.Hr(),

    # ================= Clustering Section =================
    html.H2("🤖 K-Means Cluster Analysis", style={'textAlign': 'center', 'marginTop': '40px'}),
    html.P("Grouping students based on study habits, attendance, absences, and final results.",
           style={'textAlign': 'center'}),

    html.Div([dcc.Graph(id='cluster-graph')], style={'marginTop': '20px'}),

    html.Div(id='cluster-summary', style={'marginTop': '30px', 'padding': '10px'}),

    html.Hr(),

    # ================= New Visualizations =================
    html.H2("📈 Cluster Statistical Visualization", style={'textAlign': 'center', 'marginTop': '40px'}),
    html.Div([
        dcc.Graph(id='mean-chart'),
        dcc.Graph(id='sd-chart')
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),

    html.Hr(),

    # ================= Correlation and Boxplot =================
    html.H2("📊 Correlation & Distribution Analysis", style={'textAlign': 'center', 'marginTop': '40px'}),
    html.Div([
        dcc.Graph(id='corr-heatmap'),
        dcc.Graph(id='boxplot')
    ], style={'display': 'flex', 'flexWrap': 'wrap'})
])

# ==========================================
# Callbacks
# ==========================================
@app.callback(
    Output('bar-chart', 'figure'),
    Output('pie-chart', 'figure'),
    Output('scatter-chart', 'figure'),
    Output('histogram', 'figure'),
    Output('stats-output', 'children'),
    Output('cluster-graph', 'figure'),
    Output('cluster-summary', 'children'),
    Output('mean-chart', 'figure'),
    Output('sd-chart', 'figure'),
    Output('corr-heatmap', 'figure'),
    Output('boxplot', 'figure'),
    Output('auto-insight', 'children'),
    Input('school-filter', 'value'),
    Input('age-filter', 'value')
)
def update_charts(school, agegroup):
    dff = df[(df['school'] == school) & (df['agegroup'] == agegroup)]

    if dff.empty:
        msg = "⚠️ No data available for selected filters."
        return {}, {}, {}, {}, html.H4(msg), {}, html.H4(msg), {}, {}, {}, {}, html.H4(msg)

    color_map = {'Pass': 'cornflowerblue', 'Fail': 'lightsalmon'}

    # ========== Bar Chart ==========
    bar_fig = px.bar(
        dff, x='dailystudyhours', y='finalresult', color='gradecategory',
        title='Study Hours vs Final Result', color_discrete_map=color_map
    )

    # ========== Pie Chart ==========
    pie_data = dff['gradecategory'].value_counts().reset_index()
    pie_data.columns = ['Grade', 'Count']
    pie_fig = px.pie(pie_data, names='Grade', values='Count',
                     title='Grade Distribution', color='Grade',
                     color_discrete_map=color_map)

    # ========== Scatter ==========
    scatter_fig = px.scatter(
        dff, x='attendance', y='finalresult', color='gradecategory',
        size='dailystudyhours', hover_data=['studytime', 'absences', 'age'],
        title='Attendance vs Final Result', color_discrete_map=color_map
    )

    # ========== Histogram ==========
    hist_fig = px.histogram(
        dff, x='finalresult', color='gradecategory', nbins=10,
        title='Final Result Distribution', color_discrete_map=color_map
    )

    # ========== Stats Summary ==========
    stats = html.Div([
        html.H4("📈 Stats Summary:"),
        html.P(f"Total Students: {len(dff)}"),
        html.P(f"Average Final Result: {dff['finalresult'].mean():.2f}"),
        html.P(f"Average Attendance: {dff['attendance'].mean():.2f}%"),
        html.P(f"Pass Count: {(dff['gradecategory'] == 'Pass').sum()}"),
        html.P(f"Fail Count: {(dff['gradecategory'] == 'Fail').sum()}")
    ])

    # ========== Cluster Visualization ==========
    cluster_fig = px.scatter(
        dff, x='attendance', y='finalresult', color='cluster_label',
        size='dailystudyhours', title='K-Means Student Clusters (by Performance)',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    # ========== Cluster Summary ==========
    cluster_html = [
        html.H4("🧩 Cluster Summary (Averages):"),
        html.Table([
            html.Tr([html.Th(col) for col in cluster_mean.columns])
        ] + [
            html.Tr([html.Td(row[col]) for col in cluster_mean.columns])
            for _, row in cluster_mean.iterrows()
        ], style={'border': '1px solid gray', 'padding': '8px'})
    ]

    # ========== Mean Chart ==========
    mean_fig = px.bar(
        cluster_mean, x='cluster_label',
        y=['dailystudyhours', 'attendance', 'finalresult', 'absences'],
        barmode='group', title="Cluster-wise Mean Comparison"
    )

    # ========== SD Chart ==========
    sd_fig = px.bar(
        cluster_sd, x='cluster_label',
        y=['dailystudyhours', 'attendance', 'finalresult', 'absences'],
        barmode='group', title="Cluster-wise Standard Deviation Comparison"
    )

    # ========== Correlation Heatmap ==========
    heatmap = ff.create_annotated_heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.columns),
        annotation_text=corr.round(2).values,
        colorscale='Viridis', showscale=True
    )

    # ========== Boxplot ==========
    box_fig = px.box(df, x='cluster_label', y='finalresult', color='cluster_label',
                     title='Distribution of Final Results by Cluster')

    # ========== Auto Insight ==========
    insight = f"💡 Feature most correlated with performance: **{best_feature.capitalize()}** (r = {best_corr_value:.2f})"

    return bar_fig, pie_fig, scatter_fig, hist_fig, stats, cluster_fig, cluster_html, mean_fig, sd_fig, heatmap, box_fig, insight


# ==========================================
# Run App
# ==========================================
if __name__ == '__main__':
    app.run(debug=True)
