import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from scipy.stats import ttest_ind
import statsmodels.api as sm

def validate_file_path(file_path):
    if not os.path.exists(file_path):
        return False, f"File '{file_path}' does not exist."
    if not file_path.endswith(('.csv', '.xls', '.xlsx')):
        return False, "Unsupported file format. Please provide a CSV or Excel file."
    return True, None

def validate_column_name(column_name, df):
    if column_name not in df.columns:
        return False, f"Column '{column_name}' does not exist in the dataset."
    return True, None

def validate_menu_choice(choice, valid_options):
    while choice not in valid_options:
        print(f"Invalid choice. Please select from {valid_options}.")
        choice = input("Enter your choice: ")
    return choice

def load_data(file_path):
    """
    Load data from a CSV or Excel file into a Pandas DataFrame.

    :param file_path: Path to the data file.
    :return: Pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel files.")

def summarize_data(df):
    """
    Display basic summary statistics and information about the dataset.

    :param df: Pandas DataFrame.
    """
    print("Basic Dataset Information:\n")
    print(df.info())
    print("\nSummary Statistics:\n")
    print(df.describe(include='all'))

def visualize_data(df, column_x, column_y=None, chart_type='histogram'):
    """
    Visualize data using different types of plots.

    :param df: Pandas DataFrame.
    :param column_x: Column name for the x-axis.
    :param column_y: Column name for the y-axis (optional).
    :param chart_type: Type of chart ('histogram', 'scatter', 'boxplot', 'line').
    """


    
    plt.figure(figsize=(10, 6))

    if chart_type == 'histogram':
        sns.histplot(df[column_x], kde=True)
        plt.title(f'Histogram of {column_x}')
    elif chart_type == 'scatter':
        if column_y is None:
            raise ValueError("column_y must be specified for scatter plots.")
        sns.scatterplot(data=df, x=column_x, y=column_y)
        plt.title(f'Scatter Plot: {column_x} vs {column_y}')
    elif chart_type == 'boxplot':
        sns.boxplot(data=df, x=column_x, y=column_y)
        plt.title(f'Boxplot: {column_x} vs {column_y}')
    elif chart_type == 'line':
        if column_y is None:
            raise ValueError("column_y must be specified for line plots.")
        sns.lineplot(data=df, x=column_x, y=column_y)
        plt.title(f'Line Plot: {column_x} vs {column_y}')
    else:
        raise ValueError("Unsupported chart type.")

    plt.xlabel(column_x)
    if column_y:
        plt.ylabel(column_y)
    plt.show()

def export_summary(df, output_path):
    """
    Export summary statistics to a JSON file.

    :param df: Pandas DataFrame.
    :param output_path: Path to save the JSON file.
    """
    summary = df.describe(include='all').to_dict()
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary exported to {output_path}")

def handle_missing_data(df):
    """
    Handle missing data in the dataset.
    
    :param df: Pandas DataFrame.
    :return: Modified DataFrame.
    """
    print("\nMissing Data Handling Options:")
    print("1. View missing data summary")
    print("2. Drop rows with missing values")
    print("3. Drop columns with missing values")
    print("4. Fill missing values with a specific value")
    print("5. Fill missing values with mean/median/mode")
    print("6. Return to main menu")
    
    valid_options = ['1', '2', '3', '4', '5', '6']
    choice = validate_menu_choice(input("Choose an option (1-6): "), valid_options)
    
    if choice == '1':
        print("\nMissing Data Summary:")
        print(df.isnull().sum())
        print("\nPercentage of Missing Data:")
        print((df.isnull().mean() * 100).round(2))
    elif choice == '2':
        df.dropna(inplace=True)
        print("Rows with missing values have been dropped.")
    elif choice == '3':
        df.dropna(axis=1, inplace=True)
        print("Columns with missing values have been dropped.")
    elif choice == '4':
        value = input("Enter the value to fill missing data: ")
        df.fillna(value, inplace=True)
        print("Missing values have been filled with the specified value.")
    elif choice == '5':
        method = input("Choose a method (mean/median/mode): ").lower()
        if method == 'mean':
            df.fillna(df.mean(), inplace=True)
        elif method == 'median':
            df.fillna(df.median(), inplace=True)
        elif method == 'mode':
            for column in df.columns:
                df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            print("Invalid method. Returning to the menu.")
        print(f"Missing values have been filled using the {method} method.")
    elif choice == '6':
        print("Returning to the main menu.")
    
    return df

def correlation_matrix(df):
    """
    Display the correlation matrix for numerical columns in the dataset.

    :param df: Pandas DataFrame.
    """
    print("\nCorrelation Matrix:")
    correlation = df.corr()
    print(correlation)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
def hypothesis_testing(df):
    """
    Perform a t-test between two numerical columns in the dataset.

    :param df: Pandas DataFrame.
    """
    column_x = input("Enter the first numerical column for hypothesis testing: ")
    column_y = input("Enter the second numerical column for hypothesis testing: ")

    is_valid_x, error_message_x = validate_column_name(column_x, df)
    is_valid_y, error_message_y = validate_column_name(column_y, df)

    if not is_valid_x:
        print(error_message_x)
        return
    if not is_valid_y:
        print(error_message_y)
        return

    t_stat, p_value = ttest_ind(df[column_x].dropna(), df[column_y].dropna())
    print(f"\nT-Test Results:")
    print(f"T-Statistic: {t_stat}")
    print(f"P-Value: {p_value}")

def regression_analysis(df):
    """
    Perform simple linear regression between two numerical columns.

    :param df: Pandas DataFrame.
    """
    column_x = input("Enter the independent variable (x-axis): ")
    column_y = input("Enter the dependent variable (y-axis): ")

    is_valid_x, error_message_x = validate_column_name(column_x, df)
    is_valid_y, error_message_y = validate_column_name(column_y, df)

    if not is_valid_x:
        print(error_message_x)
        return
    if not is_valid_y:
        print(error_message_y)
        return

    X = df[column_x].dropna()
    Y = df[column_y].dropna()

    X = sm.add_constant(X)  # Add constant term for regression
    model = sm.OLS(Y, X).fit()
    print("\nRegression Analysis Results:")
    print(model.summary())

def main():
    print("Academic Research Data Analysis Tool\n")
    
    file_path = input("Enter the path to your data file (CSV/Excel): ")
    is_valid, error_message = validate_file_path(file_path)
    if not is_valid:
        print(error_message)
        return

    df = load_data(file_path)
    summarize_data(df)

    while True:
        print("\nVisualization Options:")
        print("1. Histogram")
        print("2. Scatter Plot")
        print("3. Boxplot")
        print("4. Line Plot")
        print("5. Export Summary")
        print("6. Handle Missing Data")
        print("7. Bar Chart")
        print("8. Heatmap")
        print("9. Pie Chart")
        print("10.Exit ")

        valid_options = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        choice = validate_menu_choice(input("Choose an option (1-7): "), valid_options)
        
        if choice == '1':
            column_x = input("Enter the column name for the x-axis: ")
            is_valid, error_message = validate_column_name(column_x, df)
            if not is_valid:
                print(error_message)
                continue
            visualize_data(df, column_x, chart_type='histogram')
        elif choice == '2':
            column_x = input("Enter the column name for the x-axis: ")
            column_y = input("Enter the column name for the y-axis: ")
            is_valid_x, error_message_x = validate_column_name(column_x, df)
            is_valid_y, error_message_y = validate_column_name(column_y, df)
            if not is_valid_x:
                print(error_message_x)
                continue
            if not is_valid_y:
                print(error_message_y)
                continue
            visualize_data(df, column_x, column_y, chart_type='scatter')
        elif choice == '3':
            column_x = input("Enter the column name for the x-axis: ")
            column_y = input("Enter the column name for the y-axis: ")
            is_valid_x, error_message_x = validate_column_name(column_x, df)
            is_valid_y, error_message_y = validate_column_name(column_y, df)
            if not is_valid_x:
                print(error_message_x)
                continue
            if not is_valid_y:
                print(error_message_y)
                continue
            visualize_data(df, column_x, column_y, chart_type='boxplot')
        elif choice == '4':
            column_x = input("Enter the column name for the x-axis: ")
            column_y = input("Enter the column name for the y-axis: ")
            is_valid_x, error_message_x = validate_column_name(column_x, df)
            is_valid_y, error_message_y = validate_column_name(column_y, df)
            if not is_valid_x:
                print(error_message_x)
                continue
            if not is_valid_y:
                print(error_message_y)
                continue
            visualize_data(df, column_x, column_y, chart_type='line')
        elif choice == '5':
            output_path = input("Enter the path to save the summary (JSON): ")
            export_summary(df, output_path)
        elif choice == '6':  # Bar Chart
            column_x = input("Enter the column name for the x-axis: ")
            column_y = input("Enter the column name for the y-axis: ")
            is_valid_x, error_message_x = validate_column_name(column_x, df)
            is_valid_y, error_message_y = validate_column_name(column_y, df)
            if not is_valid_x:
                print(error_message_x)
                continue
            if not is_valid_y:
                print(error_message_y)
                continue
            visualize_data(df, column_x, column_y, chart_type='bar')
        elif choice == '7':  # Heatmap
            visualize_data(df, chart_type='heatmap')
        elif choice == '8':  # Pie Chart
            column_x = input("Enter the column name for the categorical data: ")
            is_valid, error_message = validate_column_name(column_x, df)
            if not is_valid:
                print(error_message)
                continue
            visualize_data(df, column_x, chart_type='pie')
       
        elif choice == '9':
            df = handle_missing_data(df)
        
        elif choice == '10':
            print("Exiting the tool. Goodbye!")
            break
        

if __name__ == "__main__":
    main()