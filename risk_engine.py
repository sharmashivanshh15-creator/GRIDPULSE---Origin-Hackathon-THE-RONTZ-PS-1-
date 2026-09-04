import os
import pandas as pd


FORECAST_PATH = "data/forecast_24h.csv"
OUTPUT_PATH = "data/risk_forecast_24h.csv"

# Configurable grid capacity for the demo.
# Keep this as a configuration parameter so it can be
# replaced with an actual grid/area capacity later.
GRID_CAPACITY_MW = 8000


def classify_risk(utilization):
    """
    Classify grid risk based on projected capacity utilization.
    """

    if utilization >= 0.95:
        return "CRITICAL"
    elif utilization >= 0.85:
        return "HIGH"
    elif utilization >= 0.70:
        return "WATCH"
    else:
        return "NORMAL"


def calculate_risk_score(utilization):
    """
    Convert utilization into a simple 0-100 risk score.
    """

    score = utilization * 100

    return round(min(score, 100), 2)


def main():

    print("=== GRID RISK ENGINE ===")

    # ---------------------------------------------------------
    # 1. Load forecast
    # ---------------------------------------------------------

    if not os.path.exists(FORECAST_PATH):
        raise FileNotFoundError(
            f"Forecast file not found: {FORECAST_PATH}\n"
            "Run forecast.py first."
        )

    forecast = pd.read_csv(FORECAST_PATH)

    forecast["timestamp"] = pd.to_datetime(
        forecast["timestamp"]
    )

    print(f"Forecast rows: {len(forecast)}")
    print(f"Grid capacity: {GRID_CAPACITY_MW:,.0f} MW")

    # ---------------------------------------------------------
    # 2. Calculate utilization
    # ---------------------------------------------------------

    forecast["capacity_utilization"] = (
        forecast["predicted_demand_mw"]
        / GRID_CAPACITY_MW
    )

    # Percentage for frontend
    forecast["utilization_percent"] = (
        forecast["capacity_utilization"] * 100
    ).round(2)

    # ---------------------------------------------------------
    # 3. Risk score
    # ---------------------------------------------------------

    forecast["risk_score"] = forecast[
        "capacity_utilization"
    ].apply(calculate_risk_score)

    # ---------------------------------------------------------
    # 4. Risk classification
    # ---------------------------------------------------------

    forecast["risk_level"] = forecast[
        "capacity_utilization"
    ].apply(classify_risk)

    # ---------------------------------------------------------
    # 5. Capacity margin
    # ---------------------------------------------------------

    forecast["capacity_margin_mw"] = (
        GRID_CAPACITY_MW
        - forecast["predicted_demand_mw"]
    ).round(2)

    # ---------------------------------------------------------
    # 6. Find peak
    # ---------------------------------------------------------

    peak_idx = forecast[
        "predicted_demand_mw"
    ].idxmax()

    peak = forecast.loc[peak_idx]

    # ---------------------------------------------------------
    # 7. Find highest-risk hours
    # ---------------------------------------------------------

    high_risk = forecast[
        forecast["risk_level"].isin(
            ["HIGH", "CRITICAL"]
        )
    ]

    # ---------------------------------------------------------
    # 8. Save results
    # ---------------------------------------------------------

    forecast.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # 9. Console report
    # ---------------------------------------------------------

    print("\n=== GRID RISK SUMMARY ===")

    print(
        f"Peak demand: "
        f"{peak['predicted_demand_mw']:,.2f} MW"
    )

    print(
        f"Peak time: "
        f"{peak['timestamp']}"
    )

    print(
        f"Peak utilization: "
        f"{peak['utilization_percent']:.2f}%"
    )

    print(
        f"Peak risk score: "
        f"{peak['risk_score']:.2f}/100"
    )

    print(
        f"Peak risk level: "
        f"{peak['risk_level']}"
    )

    print(
        f"Capacity margin at peak: "
        f"{peak['capacity_margin_mw']:,.2f} MW"
    )

    print(
        f"\nHigh/Critical hours: "
        f"{len(high_risk)}"
    )

    if len(high_risk) > 0:

        print("\n=== RISK HOURS ===")

        print(
            high_risk[
                [
                    "timestamp",
                    "predicted_demand_mw",
                    "utilization_percent",
                    "risk_score",
                    "risk_level",
                    "capacity_margin_mw",
                ]
            ].to_string(index=False)
        )

    else:

        print(
            "\nNo HIGH or CRITICAL risk hours "
            "predicted."
        )

    # ---------------------------------------------------------
    # 10. Full hourly risk table
    # ---------------------------------------------------------

    print("\n=== HOURLY RISK FORECAST ===")

    print(
        forecast[
            [
                "timestamp",
                "predicted_demand_mw",
                "utilization_percent",
                "risk_score",
                "risk_level",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nRisk forecast saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()