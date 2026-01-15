import os
import pandas as pd
import xlsxwriter

# ---------------------------------------------------
# Configuração: nomes base dos ficheiros de entrada
# (pode ser .csv ou .xlsx – o script tenta ambos)
# ---------------------------------------------------
SCENARIO_FILES = {
    "clear":  "dataset_Clear",
    "rain":   "dataset_Rain",
    "fog":    "dataset_Fog",
    "worst":  "dataset_Worst",
}

# Colunas esperadas
X_COL = "distance_m"
SNR_COLS = ["SNR_RF_dB", "SNR_FSO_dB", "SNR_THz_dB"]
EB_COLS  = ["Eb_RF_J_per_bit", "Eb_FSO_J_per_bit", "Eb_THz_J_per_bit"]
CAP_COLS = ["C_RF_bps", "C_FSO_bps", "C_THz_bps"]


def load_dataset(base_name: str) -> pd.DataFrame:
    """
    Tenta ler base_name.csv ou base_name.xlsx
    e devolve um DataFrame.
    """
    csv_path = base_name + ".csv"
    xlsx_path = base_name + ".xlsx"

    if os.path.exists(csv_path):
        print(f"  -> Loading {csv_path}")
        return pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        print(f"  -> Loading {xlsx_path}")
        return pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(
            f"Nem {csv_path} nem {xlsx_path} foram encontrados."
        )


def add_chart(workbook, df, sheet_name, title, xcol, ycols,
              cell_anchor="A1"):
    """
    Cria um gráfico de linhas e insere na folha indicada.
    """
    chart = workbook.add_chart({"type": "line"})

    # Localizar índices das colunas
    cols = df.columns.tolist()
    col_x = cols.index(xcol)

    for y in ycols:
        col_y = cols.index(y)
        chart.add_series({
            "name":       [ "Data", 0, col_y ],
            "categories": [ "Data", 1, col_x, len(df), col_x ],
            "values":     [ "Data", 1, col_y, len(df), col_y ],
        })

    chart.set_title({"name": title})
    chart.set_x_axis({"name": "Distance (m)"})
    chart.set_y_axis({"name": ""})
    chart.set_legend({"position": "bottom"})

    sheet = workbook.get_worksheet_by_name(sheet_name)
    sheet.insert_chart(cell_anchor, chart)


def build_excel_for_scenario(scen_key: str, base_name: str):
    """
    Cria um ficheiro Excel para um cenário.
    """
    print(f"\n=== Scenario: {scen_key} ===")
    df = load_dataset(base_name)

    # Ordenar por distância só para garantir
    if X_COL in df.columns:
        df = df.sort_values(by=X_COL)

    out_name = f"{scen_key}_charts.xlsx"
    print(f"  -> Creating {out_name}")

    workbook = xlsxwriter.Workbook(out_name)

    # 1) Folha de dados
    ws_data = workbook.add_worksheet("Data")

    # Escrever cabeçalho
    ws_data.write_row(0, 0, df.columns.tolist())
    # Escrever linhas
    for r, row in enumerate(df.itertuples(index=False), start=1):
        ws_data.write_row(r, 0, list(row))

    # 2) Folha de gráficos
    ws_charts = workbook.add_worksheet("Charts")
    # (deixar em branco; só vamos inserir gráficos)

    # Gráfico SNR
    add_chart(
        workbook, df, "Charts",
        f"SNR vs Distance – {scen_key.capitalize()}",
        X_COL, SNR_COLS, cell_anchor="A1"
    )

    # Gráfico Eb
    add_chart(
        workbook, df, "Charts",
        f"Energy per bit vs Distance – {scen_key.capitalize()}",
        X_COL, EB_COLS, cell_anchor="A20"
    )

    # Gráfico Capacity
    add_chart(
        workbook, df, "Charts",
        f"Capacity vs Distance – {scen_key.capitalize()}",
        X_COL, CAP_COLS, cell_anchor="A39"
    )

    workbook.close()
    print(f"  -> Saved {out_name}")


def main():
    for scen, base in SCENARIO_FILES.items():
        try:
            build_excel_for_scenario(scen, base)
        except Exception as e:
            print(f"[WARNING] Falha no cenário {scen}: {e}")


if __name__ == "__main__":
    main()
