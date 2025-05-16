from django.shortcuts import render
from django.db import connection
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import pandas as pd

# Import necessary functions from main module
from main import (
    CryptoData,
    ewma_volatility,
    ewma_covariance_matrix,
    portfolio_return,
    portfolio_volatility,
    parametric_var,
    stress_test_portfolio,
    historical_var,
    ewma_correlation_matrix
)

def crypto_list(request):
    """ Fetches cryptocurrency data from the database, calculates log returns for each entry,
    and renders the results in the 'landing.html' template.
    The function executes a raw SQL query to retrieve all columns from the 'crypto_data' table,
    along with a computed 'log_return' column, which represents the logarithmic return of the
    'close_price' compared to the previous day's close price for each ticker. The results are
    ordered by ticker and date. The log return is formatted as a string with 4 decimal places
    before being passed to the template.
    Args:
        request (HttpRequest): The HTTP request object.
    Returns:
        HttpResponse: The rendered 'landing.html' template with the cryptocurrency data."""
    sql = """
        SELECT
            *,
            CASE
                WHEN LAG(close_price) OVER (PARTITION BY ticker ORDER BY date) IS NULL THEN 0.0
                ELSE ROUND(LOG(close_price) - LOG(LAG(close_price) OVER (PARTITION BY ticker ORDER BY date)), 4)
            END AS log_return
        FROM crypto_data
        ORDER BY ticker, date
    """

    with connection.cursor() as cursor:
        cursor.execute(sql)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        data = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            # Format log_return as string with 4 decimal places
            if 'log_return' in row_dict and row_dict['log_return'] is not None:
                row_dict['log_return'] = f"{row_dict['log_return']:.4f}"
            data.append(row_dict)
    return render(request, 'myapp/landing.html', {'data': data})

@csrf_exempt
def handle_form_submission(request):
    """
    Handles the submission of a portfolio risk management form via a POST request.
    This view function processes the incoming form data, calculates various portfolio risk metrics,
    and returns the results as a JSON response. The calculations include portfolio return, EWMA-based
    covariance and correlation matrices, volatility, Value at Risk (VaR), and stress testing results.
    Args:
        request (HttpRequest): The HTTP request object containing POST data with portfolio weights.
    Returns:
        JsonResponse: A JSON response containing calculated portfolio metrics or an error message.
    Raises:
        Returns a 400 error if the request method is not POST.
        Returns a 500 error if any exception occurs during processing.
    Expected POST Data:
        - Keys starting with 'weight' representing asset weights in the portfolio.
    Response JSON Structure:
        {
            'data': {
                'portfolio_return': str,
                'ewma_covariance_matrix': dict,
                'ewma_correlation_matrix': dict,
                'ewma_volatility': dict,
                'portfolio_volatility': str,
                'parametric_var_95': str,
                'historical_var_95': str,
                'stressed_return': str,
                'stressed_scenario': str or dict,
                'sharpe_ratio': str
    """

    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)

    try:
        # Fetch crypto data
        crypto_info = CryptoData()
        data_list = crypto_info.fetch_data()  # Should return a list of dicts

        # Convert to DataFrame
        data = pd.DataFrame(data_list, columns=['ticker', 'date', 'close_price', 'log_returns'])

        # Extract weights from POST data
        weights = [
            float(value)
            for key, value in request.POST.items()
            if key.startswith('weight')
        ]

        # Calculate portfolio metrics
        port_return = portfolio_return(weights, data)
        ewma_cov = ewma_covariance_matrix(data, 0.94)
        ewma_corr = ewma_correlation_matrix(data, 0.94)
        ewma_vol = ewma_volatility(data, 0.94)
        port_vol = portfolio_volatility(weights, ewma_cov)
        param_var = parametric_var(weights, data, ewma_cov, 0.05)
        hist_var = historical_var(weights, data, 0.05)

        # Stress test (avoid redundant calls)
        stress_results = stress_test_portfolio(weights, data)
        stressed_return = stress_results.get('shocked_portfolio_return', 0.0)
        shocked_diff = stress_results.get('difference', 0.0)

        # Convert DataFrames to dicts for JSON serialization
        if isinstance(ewma_cov, pd.DataFrame):
            ewma_cov = ewma_cov.to_dict()
        if isinstance(ewma_corr, pd.DataFrame):
            ewma_corr = ewma_corr.to_dict()
        if isinstance(ewma_vol, pd.DataFrame):
            ewma_vol = ewma_vol.to_dict()
        if isinstance(shocked_diff, pd.DataFrame):
            shocked_diff = shocked_diff.to_dict()

        # Prepare response data
        response_data = {
            'portfolio_return': f"{port_return:.4f}",
            'ewma_covariance_matrix': ewma_cov,
            'ewma_correlation_matrix': ewma_corr,
            'ewma_volatility': ewma_vol,
            'portfolio_volatility': f"{port_vol:.4f}",
            'parametric_var_95': f"{param_var:.4f}",
            'historical_var_95': f"{hist_var:.4f}",
            'stressed_return': f"{stressed_return:.4f}",
            'stressed_scenario': f"{shocked_diff:.4f}" if isinstance(shocked_diff, float) else shocked_diff,
            'sharpe_ratio': f"{(port_return - 0.05) / port_vol:.4f}" if port_vol != 0 else "N/A"
        }

        # Debug output (can be removed in production)
        print(response_data['stressed_return'], response_data['stressed_scenario'])

        return JsonResponse({'data': response_data})

    except Exception as e:
        # Log the error as needed
        print(f"Error in handle_form_submission: {e}")
        return JsonResponse({'error': 'Internal server error'}, status=500)