from flask import Flask, render_template

app = Flask(__name__)

# Sector performance data
sector_performance = {
    'Financial Services': {'1wk': '0.76%', '1m': '3.88%', '3mo': '10.07%', '6mo': '18.25%', '1yr': '45.24%'},
    'Industrials': {'1wk': '0.04%', '1m': '0.46%', '3mo': '20.89%', '6mo': '41.45%', '1yr': '45.99%'},
    'Basic Materials': {'1wk': '-0.11%', '1m': '3.67%', '3mo': '13.44%', '6mo': '21.82%', '1yr': '36.73%'},
    'Consumer Defensive': {'1wk': '0.02%', '1m': '8.51%', '3mo': '24.48%', '6mo': '26.67%', '1yr': '38.01%'},
    'Consumer Cyclical': {'1wk': '-0.47%', '1m': '5.06%', '3mo': '16.33%', '6mo': '25.18%', '1yr': '42.42%'},
    'Energy': {'1wk': '-0.35%', '1m': '3.69%', '3mo': '12.79%', '6mo': '52.86%', '1yr': '37.13%'},
    'Communication Services': {'1wk': '0.63%', '1m': '11.21%', '3mo': '25.69%', '6mo': '29.44%', '1yr': '42.13%'},
    'Healthcare': {'1wk': '2.32%', '1m': '14.84%', '3mo': '26.15%', '6mo': '38.51%', '1yr': '68.93%'},
    'Technology': {'1wk': '2.50%', '1m': '8.96%', '3mo': '28.56%', '6mo': '27.57%', '1yr': '51.73%'},
    'Real Estate': {'1wk': '1.04%', '1m': '-4.03%', '3mo': '10.77%', '6mo': '36.79%', '1yr': '78.55%'},
    'Utilities': {'1wk': '2.69%', '1m': '8.46%', '3mo': '20.55%', '6mo': '50.37%', '1yr': '40.56%'},
}

@app.route('/')
def index():
    return render_template('sec.html', sector_performance=sector_performance)

if __name__ == '__main__':
    app.run(debug=True)
