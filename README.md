# CleanIV Correlation Analysis Tool

A comprehensive Python application for analyzing implied volatility correlations across equity options markets. This tool provides advanced analytics for volatility surface construction, correlation analysis, and peer-composite modeling.

## 🚀 Features

### Core Analytics
- **Implied Volatility Surface Construction**: Build clean, interpolated IV surfaces from raw options data
- **Correlation Analysis**: Compute correlations across different modes (ATM, term structure, full surface)
- **Cosine Similarity Weights**: Alternative to correlation for small n-vectors, focuses on curve shape rather than levels
- **Peer Composite Modeling**: Create weighted peer-composite volatility surfaces using correlation-weighted or cosine-weighted combinations
- **Greeks Calculation**: Full Black-Scholes Greeks computation with risk-free rates and dividend yields
- **Standard Rates**: Assumes a 4.08% risk-free rate by default, adjustable in the GUI

### Advanced Modeling
- **SVI & SABR Volatility Models**: Fit industry-standard volatility models with confidence intervals
- **ATM Pillar Analysis**: Extract and analyze at-the-money volatilities across standardized maturities
- **Relative Value Analysis**: Compare target assets against weighted peer composites using correlation weights

### Interactive GUI
- **Real-time Data Browser**: Download and analyze options data with an intuitive interface
- **Preset Management**: Save and load commonly used ticker combinations
- **Multiple Plot Types**: Volatility smiles, term structures, correlation matrices, and peer-composite surfaces
- **Dynamic Overlays**: Compare target assets with correlation-weighted peer composites

## 📦 Installation

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/froggyNews/IVCorrelation.git
cd IVCorrelation

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: save animations (MP4/GIF)
# Requires `ffmpeg`. On Windows you can install via:
# https://www.gyan.dev/ffmpeg/builds/
```

## 🎯 Quick Start

### Launch the GUI Application
```bash
python display/gui/browser.py
```

### Command Line Analysis
```python
from analysis.analysis_pipeline import *

# Download and process data
tickers = ["SPY", "QQQ", "AAPL", "MSFT"]
ingest_and_process(tickers, max_expiries=6)

# Build peer-composite surface with automatic correlation/PCA weights
target = "SPY"
peers = ["AAPL", "MSFT", "GOOGL"]
synthetic_surface, weights = build_synthetic_surface_corrweighted(target, peers)

# Compute volatility betas
betas = compute_betas("iv_atm", benchmark="SPY")
```

### Animation Controls

Interactive plots support quick visibility toggles:

* `r` – toggle Raw series
* `s` – toggle peer-composite series
* `c` – toggle confidence band
* `u` – toggle surface views

Check boxes can be enabled with `add_checkboxes` in the plotting modules for mouse-driven toggles.

## 🏗️ Project Structure

```
CleanIV_Correlation/
├── analysis/           # Core analytics engine
│   ├── analysis_pipeline.py    # Main orchestration
│   ├── correlation_builder.py  # Correlation calculations
│   ├── surface_builder.py      # IV surface construction
│   ├── peer_composite_builder.py  # Peer-composite primitives
│   ├── peer_composite_service.py  # Peer-composite orchestration
│   └── pillars.py              # ATM pillar analysis
├── data/               # Data management
│   ├── data_downloader.py      # Yahoo Finance integration
│   ├── data_pipeline.py        # Data enrichment
│   ├── ticker_groups.py        # Preset management
│   └── db_utils.py             # Database operations
├── display/            # User interface
│   ├── gui/                    # Tkinter GUI components
│   └── plotting/               # Matplotlib visualizations
├── volModel/           # Volatility modeling
│   ├── sviFit.py              # SVI model implementation
│   ├── sabrFit.py             # SABR model implementation
│   └── volModel.py            # Unified model interface
└── requirements.txt    # Dependencies
```

## 🔧 Key Components

### Analysis Pipeline
The `analysis_pipeline.py` provides a unified interface for:
- Data ingestion and enrichment
- Surface grid construction
- Correlation analysis across multiple modes
- Peer-composite construction
- Relative value analysis

### Volatility Models
- **SVI Model**: Stochastic Volatility Inspired parameterization
- **SABR Model**: Stochastic Alpha Beta Rho model
- **Polynomial Fallback**: Robust local quadratic fitting

### Database Schema
SQLite-based storage with tables for:
- Options quotes (enriched with Greeks)
- Ticker group presets
- Automated schema migrations

## 📊 Usage Examples

### GUI Preset Management
1. **Load Preset**: Select from dropdown (e.g., "Tech Giants vs SPY")
2. **Save Preset**: Enter tickers, click Save, provide name/description
3. **Analyze**: Choose plot type and run correlation analysis

### Correlation Analysis Modes
- **Underlying (`ul`)**: Based on stock price returns
- **IV ATM (`iv_atm`)**: Using at-the-money implied volatilities
- **Surface (`surface`)**: Full volatility surface correlations
- **Surface Vector (`surface_vector`)**: Treat entire surface as a flattened vector
- **Cosine ATM (`cosine_atm`)**: Shape-focused similarity for ATM curves
- **Cosine Surface (`cosine_surface`)**: Cosine similarity on full surface grids
- **Cosine UL (`cosine_ul`)**: Cosine similarity on underlying returns

- **Cosine UL Vol (`cosine_ul_vol`)**: Cosine similarity on realized vol series
- **Correlation UL Vol (`corr_ul_vol`)**: Correlation weights on realized vol series
- **Cosine Surface Vector (`cosine_surface_vector`)** and **Correlation Surface Vector (`corr_surface_vector`)**: similarity using flattened smile grids



```python
# Build peer-composite surface from correlation/PCA-derived weights
target = "SPY"
peers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
synthetic, weights = build_synthetic_surface_corrweighted(target, peers)

# Use cosine similarity for curve shape matching (ideal for ATM pillars)
weights = compute_peer_weights(target, peers, weight_mode="cosine_atm")
```

## 🎨 GUI Features

### Preset Management
- **Default Groups**: Tech Giants, Semiconductors, Financials, QQQ vs Tech
- **Custom Presets**: Save your own ticker combinations
- **Quick Load**: One-click preset application

### Plot Types
1. **Smile Plots**: K/S vs IV with model fits and confidence bands
2. **Term Structure**: ATM vol vs time to expiry
3. **Correlation Matrix**: Heatmaps with data quality indicators
4. **Peer Composite Surface**: Compare target vs weighted peer-composite smiles

### Advanced Controls
- Model selection (SVI/SABR)
- Confidence interval levels
- Time units (days/years)
- Correlation weight modes
- Pillar day customization

## 🔬 Technical Details

### Data Sources
- **Yahoo Finance**: Real-time options data via `yfinance`
- **Enrichment Pipeline**: Greeks, moneyness, ATM flagging
- **Quality Filters**: Volume, bid-ask, expiry density

### Correlation Methodologies
- **Pearson/Spearman**: Configurable correlation methods
- **Cosine Similarity**: Shape-focused similarity for small n-vectors (ideal for ATM pillars)
- **PCA Weights**: Principal component analysis for dimensionality-aware weighting
- **Rolling Windows**: Lookback period customization
- **Missing Data**: Robust handling with minimum period requirements

### Performance Optimization
- **LRU Caching**: In-memory caching for GUI responsiveness
- **Parquet Export**: Fast reload for large datasets
- **Vectorized Operations**: NumPy/Pandas optimizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for options data
- Scientific Python ecosystem (NumPy, Pandas, SciPy, Matplotlib)
- Volatility modeling research community

## 📧 Contact

For questions or collaboration opportunities, please open an issue or reach out through GitHub.

---

*Built with ❤️ for the quantitative finance community*
