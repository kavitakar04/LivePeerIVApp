"""
Data module for CleanIV_Correlation project.

This module contains data handling, downloading, and database utilities.
"""

# Import key functions for easier access
from .ticker_groups import (
    save_ticker_group,
    load_ticker_group,
    list_ticker_groups,
    delete_ticker_group,
    create_default_groups,
)
from .db_utils import get_conn, ensure_initialized, ensure_indexes
from .historical_saver import save_for_tickers
from .underlying_prices import update_underlying_prices
from .interest_rates import (
    save_interest_rate,
    load_interest_rate,
    get_default_interest_rate,
    list_interest_rates,
    delete_interest_rate,
    set_default_interest_rate,
    get_interest_rate_names,
    create_default_interest_rates,
)

__all__ = [
    "save_ticker_group",
    "load_ticker_group",
    "list_ticker_groups",
    "delete_ticker_group",
    "create_default_groups",
    "get_conn",
    "ensure_initialized",
    "ensure_indexes",
    "save_for_tickers",
    "update_underlying_prices",
    "save_interest_rate",
    "load_interest_rate",
    "get_default_interest_rate",
    "list_interest_rates",
    "delete_interest_rate",
    "set_default_interest_rate",
    "get_interest_rate_names",
    "create_default_interest_rates",
]
