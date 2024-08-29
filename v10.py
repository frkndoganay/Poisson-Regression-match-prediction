import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import font as tkfont
from tkinter import ttk

def clean_data(df):
    df.replace('-', 1, inplace=True)
    df.dropna(subset=['1', 'X', '2', 'İlk Yarı Sonucu', 'Maç Sonucu'], inplace=True)
    df['home_goals'] = df['İlk Yarı Sonucu'].apply(lambda x: int(x.split('-')[0]) if '-' in x else np.nan)
    df['away_goals'] = df['İlk Yarı Sonucu'].apply(lambda x: int(x.split('-')[1]) if '-' in x else np.nan)
    df['match_home_goals'] = df['Maç Sonucu'].apply(lambda x: int(x.split('-')[0]) if '-' in x else np.nan)
    df['match_away_goals'] = df['Maç Sonucu'].apply(lambda x: int(x.split('-')[1]) if '-' in x else np.nan)
    df.dropna(subset=['home_goals', 'away_goals', 'match_home_goals', 'match_away_goals'], inplace=True)
    return df

def build_and_fit_model(X, y):
    X_train = sm.add_constant(X)
    model = sm.GLM(y, X_train, family=sm.families.Poisson()).fit()
    return model

def evaluate_model(model, X_test, y_test):
    X_test_with_const = sm.add_constant(X_test, has_constant='add')
    predictions = model.predict(X_test_with_const)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    
    return mse, rmse, mae

def calculate_success_rate(predictions, actuals, tolerance=0.5):
    correct_predictions = np.abs(predictions - actuals) <= tolerance
    success_rate = np.mean(correct_predictions) * 100
    return success_rate

def filter_matches(df, value_1, value_x, value_2):
    values = {'1': value_1, 'X': value_x, '2': value_2}
    min_value_key = min(values, key=lambda k: float(values[k]))  # Cast to float explicitly
    other_keys = [key for key in values if key != min_value_key]
    
    tolerance = 0.10
    filtered_df = df[(df[min_value_key] == values[min_value_key]) &
                     (df[other_keys[0]] >= (values[other_keys[0]] - tolerance)) &
                     (df[other_keys[0]] <= (values[other_keys[0]] + tolerance)) &
                     (df[other_keys[1]] >= (values[other_keys[1]] - tolerance)) &
                     (df[other_keys[1]] <= (values[other_keys[1]] + tolerance))]
    
    # Combine 'İlk Yarı Sonucu' and 'Maç Sonucu' into a single string with '/'
    filtered_df['İY/MS'] = filtered_df.apply(lambda row: f"{row['İlk Yarı Sonucu']} / {row['Maç Sonucu']}", axis=1)
    
    return filtered_df[['İY/MS']]

def analyze_scores(filtered_df):
    if filtered_df.empty:
        return "Eşleşen maç bulunamadı."
    
    # Count occurrences of each score
    score_counts = filtered_df['İY/MS'].value_counts()
    
    total_matches = len(filtered_df)
    score_percentages = (score_counts / total_matches) * 100
    
    analysis_results = []
    for score, count in score_counts.items():
        percentage = score_percentages[score]
        analysis_results.append([score, count, f"{percentage:.2f}%"])
    
    return analysis_results

def main():
    file_path = 'C:/Users/Acer Aspire/Desktop/python/iddia/data.xlsx'
    df = pd.read_excel(file_path)
    df = clean_data(df)

    X = df[['1', 'X', '2']].astype(float)
    y_home_first_half = df['home_goals']
    y_away_first_half = df['away_goals']
    y_home_full_time = df['match_home_goals']
    y_away_full_time = df['match_away_goals']

    X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home_first_half, test_size=0.2, random_state=0)
    _, _, y_away_train, y_away_test = train_test_split(X, y_away_first_half, test_size=0.2, random_state=0)
    _, _, y_home_full_train, y_home_full_test = train_test_split(X, y_home_full_time, test_size=0.2, random_state=0)
    _, _, y_away_full_train, y_away_full_test = train_test_split(X, y_away_full_time, test_size=0.2, random_state=0)

    poisson_model_home_first_half = build_and_fit_model(X_train, y_home_train)
    poisson_model_away_first_half = build_and_fit_model(X_train, y_away_train)
    poisson_model_home_full_time = build_and_fit_model(X_train, y_home_full_train)
    poisson_model_away_full_time = build_and_fit_model(X_train, y_away_full_train)

    mse_home_first_half, rmse_home_first_half, mae_home_first_half = evaluate_model(poisson_model_home_first_half, X_test, y_home_test)
    mse_away_first_half, rmse_away_first_half, mae_away_first_half = evaluate_model(poisson_model_away_first_half, X_test, y_away_test)
    mse_home_full_time, rmse_home_full_time, mae_home_full_time = evaluate_model(poisson_model_home_full_time, X_test, y_home_full_test)
    mse_away_full_time, rmse_away_full_time, mae_away_full_time = evaluate_model(poisson_model_away_full_time, X_test, y_away_full_test)

    predictions_home_first_half = poisson_model_home_first_half.predict(sm.add_constant(X_test, has_constant='add'))
    predictions_away_first_half = poisson_model_away_first_half.predict(sm.add_constant(X_test, has_constant='add'))
    predictions_home_full_time = poisson_model_home_full_time.predict(sm.add_constant(X_test, has_constant='add'))
    predictions_away_full_time = poisson_model_away_full_time.predict(sm.add_constant(X_test, has_constant='add'))

    success_rate_home_first_half = calculate_success_rate(predictions_home_first_half, y_home_test)
    success_rate_away_first_half = calculate_success_rate(predictions_away_first_half, y_away_test)
    success_rate_home_full_time = calculate_success_rate(predictions_home_full_time, y_home_full_test)
    success_rate_away_full_time = calculate_success_rate(predictions_away_full_time, y_away_full_test)

    average_success_rate = np.mean([success_rate_home_first_half, success_rate_away_first_half, success_rate_home_full_time, success_rate_away_full_time])

    def round_goals(predicted_goals_home, predicted_goals_away):
        def round_single_goal(goal_value, is_home_team):
            if goal_value > 2.15:
                return 3
            elif goal_value > 1.30:
                return 2
            elif (goal_value > 0.60 and is_home_team) or (goal_value > 0.40 and not is_home_team):
                return 1
            else:
                return 0

        home_goals = round_single_goal(predicted_goals_home, True)
        away_goals = round_single_goal(predicted_goals_away, False)
        
        return home_goals, away_goals

    def predict():
        try:
            value_1 = float(entry_1.get())
            value_x = float(entry_x.get())
            value_2 = float(entry_2.get())

            input_values = pd.DataFrame([[value_1, value_x, value_2]], columns=['1', 'X', '2'])
            input_values = sm.add_constant(input_values, has_constant='add')

            predicted_home_goals_first_half = poisson_model_home_first_half.predict(input_values)[0]
            predicted_away_goals_first_half = poisson_model_away_first_half.predict(input_values)[0]
            predicted_home_goals_full_time = poisson_model_home_full_time.predict(input_values)[0]
            predicted_away_goals_full_time = poisson_model_away_full_time.predict(input_values)[0]

            # Apply the custom rounding logic
            rounded_home_goals_first_half, rounded_away_goals_first_half = round_goals(predicted_home_goals_first_half, predicted_away_goals_first_half)
            rounded_home_goals_full_time, rounded_away_goals_full_time = round_goals(predicted_home_goals_full_time, predicted_away_goals_full_time)

            result_text = (f"\nTahmin Edilen İlk Yarı Maç Skoru: {rounded_home_goals_first_half} - {rounded_away_goals_first_half}\n"
                        f"Tahmin Edilen Maç Sonucu: {rounded_home_goals_full_time} - {rounded_away_goals_full_time}\n")
            
            # Analyze filtered results
            filtered_df = filter_matches(df, value_1, value_x, value_2)
            analysis_results = analyze_scores(filtered_df)

            # Clear the Treeview
            for row in tree.get_children():
                tree.delete(row)

            # Insert the analysis results into the Treeview
            for result in analysis_results:
                tree.insert("", tk.END, values=result)
                
            # Show the result text in the result box
            result_box.config(state=tk.NORMAL)
            result_box.delete(1.0, tk.END)
            result_box.insert(tk.END, result_text)
            result_box.config(state=tk.DISABLED)
        
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli bir sayı girin.")

    # Create the GUI
    root = tk.Tk()
    root.title("Futbol Maç Tahmincisi")

    # Define custom font
    custom_font = tkfont.Font(family="Helvetica", size=12)

    # Create and place widgets
    tk.Label(root, text="Ev Sahibi Kazanma Oranı (1):", font=custom_font).pack(pady=5)
    entry_1 = tk.Entry(root, font=custom_font)
    entry_1.pack(pady=5)

    tk.Label(root, text="Beraberlik Oranı (X):", font=custom_font).pack(pady=5)
    entry_x = tk.Entry(root, font=custom_font)
    entry_x.pack(pady=5)

    tk.Label(root, text="Deplasman Kazanma Oranı (2):", font=custom_font).pack(pady=5)
    entry_2 = tk.Entry(root, font=custom_font)
    entry_2.pack(pady=5)

    tk.Button(root, text="Tahmin Et", command=predict, font=custom_font).pack(pady=10)

    # Result box
    result_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=6, width=60, font=custom_font, state=tk.DISABLED)
    result_box.pack(pady=10)

    # Create Treeview
    tree = ttk.Treeview(root, columns=("score", "count", "percentage"), show='headings')
    tree.heading("score", text="Skor", anchor=tk.CENTER)
    tree.heading("count", text="Sayı", anchor=tk.CENTER)
    tree.heading("percentage", text="Yüzde", anchor=tk.CENTER)

    # Center align column contents
    tree.column("score", anchor=tk.CENTER, width=200)
    tree.column("count", anchor=tk.CENTER, width=120)
    tree.column("percentage", anchor=tk.CENTER, width=120)

    tree.pack(pady=10, fill=tk.X)

    root.mainloop()

if __name__ == "__main__":
    main()
