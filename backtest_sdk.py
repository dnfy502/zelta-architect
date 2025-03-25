from untrade.client import Client
from pprint import pprint


def perform_backtest(csv_file_path):
    # Create an instance of the untrade client
    client = Client()

    # Perform backtest using the provided CSV file path
    result = client.backtest(
        file_path=csv_file_path,
        #leverage = 1,
        jupyter_id="dnfyarchitect",  # the one you use to login to jupyter.untrade.io
        #result_type="",
    )
    # result_type can be one of the following: Y, M, QS

    return result


if __name__ == "__main__":
    csv_file_path = "results.csv"
    backtest_result = perform_backtest(csv_file_path)
    test = list(backtest_result)
    print(backtest_result)
    for value in test:
        print(value)

