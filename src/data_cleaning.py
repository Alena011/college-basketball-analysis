import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def clean_data(df):
    """Очистка данных"""
    df_clean = df.dropna().drop_duplicates()

    # Исправлено: 'COMF' → 'CONF' (верное название столбца)
    df_clean = pd.get_dummies(df_clean, columns=['CONF'], prefix='conf')

    # Исправлено: дублирование 'ADJDE' → 'ADJOE'
    numerical_cols = ['ADJOE', 'ADJDE', 'W']
    scaler = MinMaxScaler()
    df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])

    return df_clean


if __name__ == "__main__":
    input_path = "../data/cbb.csv"  # Две точки - подняться на уровень выше
    output_path = "../data/cleaned_cbb.csv"

    df = pd.read_csv(input_path)
    df_clean = clean_data(df)
    df_clean.to_csv(output_path, index=False)
    print("Данные успешно обработаны и сохранены!")