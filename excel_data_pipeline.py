import argparse
from pathlib import Path

import pandas as pd


def process_sheet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    group_columns = list(df.columns)
    grouped = (
        df.groupby(group_columns, dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    grouped["count_x4"] = grouped["count"] * 4
    return grouped


def process_workbook(input_path: Path, output_path: Path) -> None:
    workbook = pd.read_excel(input_path, sheet_name=None)
    output_sheets = {}

    for sheet_name, sheet_df in workbook.items():
        cleaned = process_sheet(sheet_df)
        output_sheets[sheet_name] = cleaned

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, cleaned_df in output_sheets.items():
            cleaned_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Processed {len(output_sheets)} sheets and wrote output to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate all sheets in an Excel workbook and add row-count metadata."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input Excel workbook.",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        help="Path to the cleaned output Excel workbook. If omitted, adds '_deduped' to the input filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_file
    output_path = args.output_file or input_path.with_name(
        input_path.stem + "_deduped" + input_path.suffix
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    process_workbook(input_path, output_path)


if __name__ == "__main__":
    main()
